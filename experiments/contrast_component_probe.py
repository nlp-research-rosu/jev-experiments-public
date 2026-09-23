"""Inference-only component, representation and evidence-source probes on frozen contrasts."""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.semantic_contrasts import prediction_view, relation_metrics, summarize, validate_suite
from openjev.judgment_cli import load_judgment_engine, read_json

HEAD_NAMES = ("compatibility.weight", "binary.weight", "binary.bias")
RECORD_FIELDS = ("evaluated_at", "authority", "baseline", "current_record_snapshots")
SOURCE_FIELDS = {
    "expressed_intent": ("request", "assistant_messages"),
    "claimed_done": ("request", "assistant_messages"),
    "invocation_shown": ("request", "audit_scope", "audit_events"),
    "completion_confirmed": ("request", "audit_scope", "audit_events"),
    "effective_now": ("request", "audit_scope", "audit_events"),
    "outcome_status": ("request", "audit_scope", "audit_events"),
    "permitted": ("request", "policy"),
    "explicit_change_claim": ("evaluated_at", "baseline", "message"),
    "different_destination": ("evaluated_at", "transaction", "baseline", "requested_destination"),
    "record_confirms_change": RECORD_FIELDS,
    "record_status": RECORD_FIELDS,
    "use_permitted": (*RECORD_FIELDS, "requested_destination", "use_policy", "transaction"),
    "operational_change_signal": ("evaluated_at", "transaction", "baseline", "message", "requested_destination"),
}


def context_groups(request, *, filtered):
    """Known-fixture semantic field selection, never label-dependent or a general retriever."""
    groups = {}
    for qid, question in request["questions"].items():
        if qid not in SOURCE_FIELDS:
            raise ValueError("unreviewed question predicate")
        fields = SOURCE_FIELDS[qid]
        if any(field not in request["state"] for field in fields):
            raise ValueError("required evidence field absent")
        if fields not in groups:
            state = {key: request["state"][key] for key in fields} if filtered else request["state"]
            groups[fields] = {"state": copy.deepcopy(state), "questions": {}}
        groups[fields]["questions"][qid] = copy.deepcopy(question)
    return list(groups.values())


def state_as_json_text(request):
    return {
        "state": json.dumps(request["state"], ensure_ascii=False, sort_keys=True),
        "questions": copy.deepcopy(request["questions"]),
    }


def snapshot_heads(model):
    parameters = dict(model.named_parameters())
    return {name: parameters[name].detach().cpu().clone() for name in HEAD_NAMES}


def restore_heads(model, heads):
    if set(heads) != set(HEAD_NAMES):
        raise ValueError("incomplete head state")
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in HEAD_NAMES:
            if heads[name].shape != parameters[name].shape:
                raise ValueError("head shape mismatch")
            parameters[name].copy_(heads[name].to(parameters[name].device))


@contextmanager
def model_components(model, heads, *, adapters):
    """Temporary in-memory intervention, restored even if evaluation fails."""
    previous = snapshot_heads(model)
    context = nullcontext() if adapters else model.backbone.disable_adapter()
    try:
        restore_heads(model, heads)
        with context:
            yield model
    finally:
        restore_heads(model, previous)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate(engine, suite, mode, out):
    rows, lookup, max_tokens = [], {}, 0
    for index, case in enumerate(suite["cases"]):
        if mode == "full":
            requests = [case["request"]]
        elif mode == "json_text":
            request = state_as_json_text(case["request"])
            if json.loads(request["state"]) != case["request"]["state"]:
                raise RuntimeError("representation lost information")
            requests = [request]
        else:
            requests = context_groups(case["request"], filtered=mode == "relevant_grouped")
        answers, responses = {}, []
        for request in requests:
            response = engine.evaluate(request, details=True, cached=False)
            if set(answers) & set(response["answers"]):
                raise RuntimeError("duplicate grouped question")
            answers.update(response["answers"])
            max_tokens = max(max_tokens, response["usage"]["max_unit_tokens"])
            responses.append(response)
        if set(answers) != set(case["expected"]):
            raise RuntimeError("incomplete grouped response")
        for qid, question in case["request"]["questions"].items():
            pred = prediction_view(question, case["expected"][qid], answers[qid])
            rows.append(
                {
                    "case_id": case["id"],
                    "question_id": qid,
                    "family_id": case["family_id"],
                    "domain": case["domain"],
                    "variant": case["variant"],
                    "prediction": pred,
                }
            )
            lookup[case["id"], qid] = pred
        with (out / "responses.jsonl").open("a") as stream:
            stream.write(json.dumps({"case_id": case["id"], "responses": responses}, allow_nan=False) + "\n")
        if (index + 1) % 20 == 0:
            print(out.name, f"{index + 1}/{len(suite['cases'])}", flush=True)
    relations = [{**r, "metrics": relation_metrics(r, lookup)} for r in suite["relations"]]
    write(out / "predictions.json", rows)
    write(out / "relations.json", relations)
    return {"summary": summarize(rows, relations), "max_unit_tokens": max_tokens}, rows


def compare_predictions(a, b):
    before = {(r["case_id"], r["question_id"]): r["prediction"] for r in a}
    changes, largest = 0, 0.0
    if len(a) != len(b):
        raise ValueError("prediction populations differ")
    for row in b:
        old, new = before[row["case_id"], row["question_id"]], row["prediction"]
        changes += old["predicted"] != new["predicted"]
        largest = max(
            largest, max(abs(value - new["probabilities"][key]) for key, value in old["probabilities"].items())
        )
    return {"questions": len(b), "label_changes": changes, "max_probability_difference": largest}


def run(args, report):
    manifest = read_json(args.suite / "manifest.json")
    if sha(args.suite / "suite.json") != manifest["suite_sha256"]:
        raise ValueError("frozen fixture hash mismatch")
    suite = read_json(args.suite / "suite.json")
    validate_suite(suite)
    report["suite_sha256"] = manifest["suite_sha256"]
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = load_judgment_engine(backend="fla")
    initial_heads = snapshot_heads(base.model)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    engine = load_judgment_engine(checkpoint=args.checkpoint, backend="fla", unit_batch_size=4, max_input_tokens=4096)
    trained_heads = snapshot_heads(engine.model)
    report["checkpoint_id"] = engine.model_id
    report["head_changes"] = {
        name: {
            "initial_norm": initial_heads[name].norm().item(),
            "trained_norm": trained_heads[name].norm().item(),
            "difference_norm": (trained_heads[name] - initial_heads[name]).norm().item(),
        }
        for name in HEAD_NAMES
    }
    conditions = [
        ("base_body_base_heads", False, initial_heads, "full"),
        ("adapted_body_trained_heads", True, trained_heads, "full"),
        ("base_body_trained_heads", False, trained_heads, "full"),
        ("adapted_body_base_heads", True, initial_heads, "full"),
        ("trained_full_grouped", True, trained_heads, "full_grouped"),
        ("trained_relevant_grouped", True, trained_heads, "relevant_grouped"),
        ("trained_json_text", True, trained_heads, "json_text"),
    ]
    report["conditions"] = {}
    predictions = {}
    for name, adapters, heads, context_mode in conditions:
        out = args.output / name
        out.mkdir()
        engine.model_id = f"openjev-judgment-v0.2/diagnostic/{name}"
        print("Condition", name, flush=True)
        with model_components(engine.model, heads, adapters=adapters):
            result, rows = evaluate(engine, suite, context_mode, out)
        predictions[name] = rows
        if name in ("base_body_base_heads", "adapted_body_trained_heads"):
            old_variant = "untrained" if name == "base_body_base_heads" else "trained"
            old = read_json(Path("reports/semantic-contrasts-v1") / (old_variant + "-predictions.json"))
            reproduction = compare_predictions(old, rows)
            result["saved_diagonal_reproduction"] = reproduction
            if reproduction["label_changes"] or reproduction["max_probability_difference"] > 1e-5:
                raise RuntimeError("component experiment failed to reproduce its original diagonal control")
        report["conditions"][name] = {"adapters": adapters, "context": context_mode, **result}
        write(args.output / "report.json", report)
    report["comparisons"] = {
        "grouping_vs_original": compare_predictions(
            predictions["adapted_body_trained_heads"], predictions["trained_full_grouped"]
        ),
        "filtering_vs_matched_grouping": compare_predictions(
            predictions["trained_full_grouped"], predictions["trained_relevant_grouped"]
        ),
        "json_text_vs_original": compare_predictions(
            predictions["adapted_body_trained_heads"], predictions["trained_json_text"]
        ),
    }
    for name, value in snapshot_heads(engine.model).items():
        torch.testing.assert_close(value, trained_heads[name], atol=0, rtol=0)
    if engine.model.backbone.get_model_status().enabled is not True:
        raise RuntimeError("adapter state was not restored")
    report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite", default=Path("data/semantic-contrasts-v1"), type=Path)
    parser.add_argument("--checkpoint", default=Path("checkpoints/judgment-full-v0.2/final"), type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output")
    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path("experiments/semantic_contrasts.py"),
                *Path("src/openjev").glob("judgment*.py"),
            ]
        },
        "packages": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "fla-core")},
        "scope": "Retrospective inference interventions on the inspected frozen diagnostic. No optimizer, training, prompt tuning or checkpoint writes. Head/body swaps are coadaptation probes, not unique causal attribution.",
        "filtering_policy": SOURCE_FIELDS,
        "filtering_limit": "Known-predicate source selection; no gold-label selection. Removing fields changes both evidence interference and input length. Full-grouped is the matched batching control.",
        "representation_limit": "Same record passed as a JSON string instead of an object; information round-trips exactly, but tokenizer length/escaping changes.",
    }
    write(args.output / "report.json", report)
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
