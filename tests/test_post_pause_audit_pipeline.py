"""Independent CPU checks of the paused scaling run; no test split is read."""

import hashlib
import json
import math
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig

from experiments.contrast_scaling_training import backward_family_loss, make_family_schedule, prepare_canonical_replay
from experiments.judgment_pipeline import optimizer_for
from openjev.judgment_model import MODEL_ID, REVISION, JudgmentEngine, JudgmentModel
from openjev.judgment_training import PreparedBundle, TrainingGroup, prepare_bundle
from openjev.judgments import compile_request, render_unit_messages

ROOT = Path(__file__).resolve().parents[1]


def small_model():
    torch.manual_seed(903)
    config = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000., "partial_rotary_factor": .5,
                         "mrope_section": [1, 1, 2]},
        pad_token_id=0,
    )
    return JudgmentModel(Qwen3_5ForCausalLM(config), kernel_backend="reference")


class PipelineAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_analytic_six_component_gradient_with_complete_candidate_groups(self):
        """Hand derivatives, independent of both production loss implementations."""
        values, specs = [], []

        def bundle(identity, shapes, origin):
            prompts, kinds, groups = [], [], []
            for j, (kind, width) in enumerate(shapes):
                ids = list(range(len(values), len(values) + width))
                values.extend(math.sin(i + .2) * 2 for i in ids)
                # Candidate storage and grouping intentionally differ.
                positions = tuple(reversed(range(len(prompts), len(prompts) + width)))
                prompts.extend([[i] for i in ids])
                kinds.extend([int(kind == "noul")] * width)
                criteria = {name: name for name in ["zeta", "alpha", "mu", "beta", "eta"][:width]} if kind == "choice" else list(range(width))
                target_index = (j + 1) % (2 if kind == "noul" else width)
                target = ({"truth": bool(target_index)} if kind == "noul" else
                          {"choice": list(criteria)[target_index]} if kind == "choice" else {"level_index": target_index})
                group = TrainingGroup(kind, positions, criteria, target)
                groups.append(group)
                specs.append((origin, kind, list(reversed(ids)), target_index))
            return PreparedBundle(identity, "audit", prompts, kinds, groups, [])

        family = bundle("family", [("noul", 1)] * 12 + [("choice", w) for w in (3, 4, 5, 3)] + [("score", w) for w in (2, 3, 4, 5)], "new")
        replay = {kind: bundle(kind, [(kind, width)], "old") for kind, width in (("noul", 1), ("choice", 4), ("score", 5))}
        counts = Counter((o, k) for o, k, _, _ in specs)
        expected_gradient = [0.] * len(values)
        expected_loss = 0.
        for origin, kind, ids, target in specs:
            weight = 1 / (6 * counts[origin, kind])
            scores = [values[i] for i in ids]
            if kind == "noul":
                p = 1 / (1 + math.exp(-scores[0]))
                loss = -math.log(p if target else 1-p)
                expected_gradient[ids[0]] = weight * (p-target)
            else:
                denominator = sum(math.exp(z) for z in scores)
                probabilities = [math.exp(z)/denominator for z in scores]
                loss = -math.log(probabilities[target])
                for j, idx in enumerate(ids):
                    expected_gradient[idx] = weight * (probabilities[j] - int(j == target))
            expected_loss += weight * loss

        class Table(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor(values, dtype=torch.float64))

            def score_prompts(self, prompts, kinds, *, unit_batch_size):
                return self.logits[[p[0] for p in prompts]]

        for max_units, batch_size in ((5, 1), (8, 4), (12, 4)):
            with self.subTest(max_units=max_units, batch_size=batch_size):
                model = Table()
                result = backward_family_loss(model, family, replay, max_units=max_units, unit_batch_size=batch_size)
                self.assertAlmostEqual(result["loss"], expected_loss, places=13)
                torch.testing.assert_close(model.logits.grad, torch.tensor(expected_gradient, dtype=torch.float64), atol=1e-15, rtol=1e-13)
                self.assertEqual(result["counts"], {f"{o}/{k}": c for (o, k), c in counts.items()})

    def test_hybrid_logits_and_gradients_match_unpadded_independent_forwards(self):
        model = small_model()
        model.add_lora(rank=2, alpha=4, gradient_checkpointing=True)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, .015)
        model.train()
        prompts = [[0], [0, 3, 4], [0] + [(i * 7) % 96 for i in range(1, 65)], [0] + [(i * 11) % 96 for i in range(1, 142)]]
        kinds = [1, 0, 1, 0]
        parameters = [p for p in model.parameters() if p.requires_grad]
        expected = []
        for p, kind in zip(prompts, kinds):
            h = model.backbone(input_ids=torch.tensor([p]), use_cache=False).last_hidden_state[0, -1].float()
            head = model.binary if kind else model.compatibility
            expected.append(head(h).reshape(()))
        reference = torch.stack(expected)
        reference_grad = torch.autograd.grad(reference.square().sum(), parameters)
        actual = model.score_prompts(prompts, kinds, unit_batch_size=4)
        actual_grad = torch.autograd.grad(actual.square().sum(), parameters)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
        for p, q in zip(actual_grad, reference_grad):
            torch.testing.assert_close(p, q, atol=2e-5, rtol=2e-4)
        model.eval()
        with torch.inference_mode():
            evaluated = model.score_prompts(prompts, kinds, unit_batch_size=4)
            cached, _ = model.score_cached(prompts, kinds, unit_batch_size=3)
        torch.testing.assert_close(evaluated, actual, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(cached, actual, atol=2e-6, rtol=2e-5)
        self.assertTrue(all(not p.requires_grad for n, p in model.named_parameters() if ".lora_" not in n and not n.startswith(("binary.", "compatibility."))))
        optimizer = optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]}, {id(p) for p in parameters})

    def test_real_prepared_samples_keep_definitions_candidate_order_and_labels(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        engine = JudgmentEngine(None, tokenizer, max_input_tokens=1536)
        selected = []
        with (ROOT / "data/contrast-scaling-v1/train.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                if len(selected) < 10:
                    selected.append(record)
                else:
                    break
        self.assertEqual(len({r["provenance"]["category"] for r in selected}), 10)
        for record in selected:
            prepared = prepare_bundle(engine, record)
            self.assertEqual(len(prepared.groups), 20)
            for example, group in zip(record["examples"], prepared.groups):
                compiled = compile_request({"state": example["state"], "questions": {"q": example["question"]}})
                for position, unit in zip(group.indices, compiled.units):
                    self.assertEqual(unit.payload["state"], example["state"])
                    state = unit.payload["state"]
                    self.assertIn("choice_definitions", state)
                    self.assertIn("score_levels", state)
                    self.assertEqual(prepared.kinds[position], int(group.primitive == "noul"))
                    ids = tokenizer.apply_chat_template(render_unit_messages(unit), tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                    self.assertEqual(prepared.prompts[position], ids)
                    self.assertLessEqual(len(ids), 1536)
                    self.assertEqual(set(unit.payload), {"state", "question", "binary_criteria" if group.primitive == "noul" else "criterion"})
                self.assertEqual(group.target, example["target"])
                if group.primitive == "choice":
                    self.assertEqual([u.payload["criterion"]["name"] for u in compiled.units], list(example["question"]["criteria"]))
                elif group.primitive == "score":
                    self.assertEqual([u.payload["criterion"]["description"] for u in compiled.units], example["question"]["criteria"])

    def test_all_frozen_training_schema_labels_and_duplicate_input_consistency(self):
        manifest = json.loads((ROOT / "data/contrast-scaling-v1/manifest.json").read_text())
        identities, categories, seen = [], Counter(), {}
        with (ROOT / "data/contrast-scaling-v1/train.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                identities.append(row["id"])
                categories[row["provenance"]["category"]] += 1
                self.assertEqual(row["provenance"]["assigned_split"], "train")
                self.assertEqual(len(row["examples"]), 20)
                by_case = defaultdict(dict)
                for example in row["examples"]:
                    by_case[example["case_id"]][example["question_id"]] = example
                    question, state, target = example["question"], example["state"], example["target"]
                    kind = question["type"]
                    self.assertEqual(set(target), {{"noul": "truth", "choice": "choice", "score": "level_index"}[kind]})
                    if kind == "noul":
                        self.assertIs(type(target["truth"]), bool)
                    elif kind == "choice":
                        self.assertIn(target["choice"], question["criteria"])
                        self.assertEqual(state["choice_definitions"], question["criteria"])
                    else:
                        self.assertIs(type(target["level_index"]), int)
                        self.assertTrue(0 <= target["level_index"] < len(question["criteria"]))
                        self.assertEqual(state["score_levels"], question["criteria"])
                    request = {"state": state, "questions": {"answer": question}}
                    digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).digest()
                    label = json.dumps(target, sort_keys=True)
                    if digest in seen:
                        self.assertEqual(seen[digest], label)
                    seen[digest] = label
                self.assertEqual(len(by_case), 4)
                for case in by_case.values():
                    self.assertEqual(set(case), {"n1", "n2", "n3", "c1", "s1"})
                    self.assertTrue(all(e["state"] == case["c1"]["state"] for e in case.values()))
        self.assertEqual(identities, manifest["ordered_family_ids"])
        self.assertEqual(len(identities), 5000)
        self.assertEqual(len(seen), 100000)
        self.assertEqual(sorted(categories.values()), [500] * 10)

    def test_real_canonical_replay_pool_and_equal_position_schedule(self):
        # A compile-only engine avoids retokenizing 4800 old questions; the prior
        # test covers the pinned tokenizer independently.
        class CompileOnly:
            def prepare(self, request):
                compiled = compile_request(request)
                return compiled, [[i+1] for i in range(len(compiled.units))], [u.readout_kind for u in compiled.units]

        prepared, pools = prepare_canonical_replay(CompileOnly(), ROOT / "data/processed-v0.2/train.jsonl")
        seen, sources = set(), Counter()
        original = sorted((json.loads(line) for line in (ROOT / "data/processed-v0.2/train.jsonl").read_text().splitlines()), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        for record in original:
            source = record["provenance"]["dataset"]
            if sources[source] >= 800:
                continue
            sources[source] += 1
            identity = "old/" + record["id"]
            seen.add(identity)
            self.assertEqual(prepared[identity].groups[0].target, record["examples"][0]["target"])
            self.assertEqual(prepared[identity].groups[0].criteria, record["examples"][0]["question"]["criteria"])
            self.assertEqual(len(prepared[identity].groups), 1)
        self.assertEqual(seen, set(prepared))
        self.assertTrue(all(n == 800 for n in sources.values()))
        families = [str(i) for i in range(1000)]
        primary = make_family_schedule(families, pools, 1000)
        repeated = make_family_schedule(families, pools, 1000, repeat_size=200)
        self.assertEqual([r["replay"] for r in primary], [r["replay"] for r in repeated])
        for kind, by_source in pools.items():
            counts = Counter(item["source"] for row in primary for item in row["replay"] if item["primitive"] == kind)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_saved_new_validation_metrics_reconstruct_from_raw_logits(self):
        suite = json.loads((ROOT / "data/contrast-scaling-v1/validation/suite.json").read_text())
        cases = {c["id"]: c for c in suite["cases"]}
        self._assert_archived_metrics(cases, "new", 800)

    def test_saved_broad_validation_metrics_reconstruct_from_original_labels(self):
        records = sorted((json.loads(line) for line in (ROOT / "data/processed-v0.2/validation.jsonl").read_text().splitlines()), key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        counts, cases = Counter(), {}
        for row in records:
            source = row["provenance"]["dataset"]
            if counts[source] >= 50:
                continue
            counts[source] += 1
            self.assertEqual(row["provenance"]["assigned_split"], "validation")
            example = row["examples"][0]
            question = example["question"]
            key = {"noul": "truth", "choice": "choice", "score": "level_index"}[question["type"]]
            cases[row["id"]] = {"id": row["id"], "expected": {source: example["target"][key]}, "request": {"questions": {source: question}}}
        self.assertEqual(len(cases), 300)
        self._assert_archived_metrics(cases, "broad", 300)

    def _assert_archived_metrics(self, cases, suite_name, questions):
        base = ROOT / "reports/contrast-scaling-v1/mid-gpu-validation"
        for step in (0, 200, 400, 600, 800, 1000, 1692):
            location = base / f"step-{step:04d}" / suite_name
            predictions = json.loads((location / "predictions.json").read_text())
            lookup = {(r["case_id"], r["question_id"]): r["prediction"] for r in predictions}
            expected_keys = {(c["id"], q) for c in cases.values() for q in c["expected"]}
            self.assertEqual(set(lookup), expected_keys)
            self.assertEqual(len(predictions), len(expected_keys))
            count = 0
            for line in (location / "responses.jsonl").read_text().splitlines():
                record = json.loads(line)
                case = cases[record["case_id"]]
                for qid, answer in record["response"]["answers"].items():
                    target, kind = case["expected"][qid], case["request"]["questions"][qid]["type"]
                    saved = lookup[case["id"], qid]
                    label = ("true" if target else "false") if kind == "noul" else str(target)
                    self.assertEqual(saved["expected"], label)
                    if kind == "noul":
                        z = torch.tensor(answer["details"]["raw_logit"], dtype=torch.float64)
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, z.new_tensor(float(target))).item()
                        probs = {"false": (-z).sigmoid().item(), "true": z.sigmoid().item()}
                    else:
                        names, raw = zip(*answer["details"]["raw_logits"].items())
                        z = torch.tensor(raw, dtype=torch.float64)
                        loss = torch.nn.functional.cross_entropy(z, torch.tensor(names.index(label))).item()
                        probs = dict(zip(names, z.softmax(0).tolist()))
                    self.assertAlmostEqual(saved["nll"], loss, places=11)
                    for name, p in probs.items():
                        self.assertAlmostEqual(saved["probabilities"][name], p, places=14)
                    brier = sum((p - int(k == label)) ** 2 for k, p in probs.items())
                    self.assertAlmostEqual(saved["brier"], brier, places=13)
                    maximum = max(probs.values())
                    top = [k for k, p in probs.items() if abs(p-maximum) <= 1e-12]
                    self.assertEqual(saved["correct"], len(top) == 1 and top[0] == label)
                    self.assertAlmostEqual(saved["max_probability"], maximum, places=14)
                    if kind == "score":
                        mean = sum(int(k) * p for k, p in probs.items())
                        self.assertAlmostEqual(saved["score_mean"], mean, places=13)
                        self.assertAlmostEqual(answer["score"], mean, places=13)
                        self.assertAlmostEqual(saved["score_absolute_error"], abs(mean-target), places=13)
                    count += 1
            self.assertEqual(count, questions)

    def test_every_validation_question_has_same_input_in_bundle_and_whole_request(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        engine = JudgmentEngine(None, tokenizer, max_input_tokens=1536)
        suite = json.loads((ROOT / "data/contrast-scaling-v1/validation/suite.json").read_text())
        count = 0
        for case in suite["cases"]:
            compiled, prompts, kinds = engine.prepare(case["request"])
            self.assertEqual([q.path[0] for q in compiled.questions], list(case["request"]["questions"]))
            for q in compiled.questions:
                qid = q.path[0]
                question = case["request"]["questions"][qid]
                y = case["expected"][qid]
                key = {"noul": "truth", "choice": "choice", "score": "level_index"}[q.primitive]
                single = prepare_bundle(engine, {"id": "audit", "examples": [{"state": case["request"]["state"], "question": question, "target": {key: y}}]})
                self.assertEqual([prompts[i] for i in q.unit_indices], single.prompts)
                self.assertEqual([kinds[i] for i in q.unit_indices], single.kinds)
                for unit_index in q.unit_indices:
                    state = compiled.units[unit_index].payload["state"]
                    self.assertEqual(state["choice_definitions"], case["request"]["questions"]["c1"]["criteria"])
                    self.assertEqual(state["score_levels"], case["request"]["questions"]["s1"]["criteria"])
                count += 1
        self.assertEqual(count, 800)

    def test_tiny_production_checkpoint_optimizer_and_next_update(self):
        model = small_model()
        model.add_lora(rank=2, alpha=4, gradient_checkpointing=True)
        optimizer = optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))

        def update(current, optim):
            current.train()
            optim.zero_grad(set_to_none=True)
            z = current.score_prompts([[1, 2, 3, 0], [1, 2, 4, 5, 6]], [1, 0])
            (z-torch.tensor([1., -1.])).square().sum().backward()
            torch.nn.utils.clip_grad_norm_([p for p in current.parameters() if p.requires_grad], 1.)
            optim.step()

        update(model, optimizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint"
            model.save_checkpoint(path, optimizer=optimizer, training_state={"step": 1, "next_step": 1})
            update(model, optimizer)
            actual = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
            # Reconstruct a causal module with the exact seeded frozen body; no
            # production resume helper participates in this comparison.
            torch.manual_seed(903)
            config = model.backbone.base_model.model.config
            restored, _ = JudgmentModel.load_checkpoint(path, base_factory=lambda: Qwen3_5ForCausalLM(config), trainable=True)
            restored_optimizer = optimizer_for(restored, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
            payload = torch.load(path / "training.pt", map_location="cpu", weights_only=False)
            restored_optimizer.load_state_dict(payload["optimizer"])
            torch.set_rng_state(payload["torch_rng"])
            self.assertEqual(payload["state"], {"step": 1, "next_step": 1})
            update(restored, restored_optimizer)
            for name, parameter in restored.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter, actual[name], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
