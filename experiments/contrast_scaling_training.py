"""Bounded, family-preserving runner for the contrast data-scaling study.

This module deliberately contains no data authoring and never evaluates a test
suite while training.  It prepares every member of a declared family, gives
the six origin/primitive components equal weight, and preserves the exact
optimizer and RNG state needed to branch the 200-family repeat control.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from experiments.judgment_pipeline import optimizer_for
from openjev.judgment_cli import load_judgment_engine
from openjev.judgment_model import frozen_digest
from openjev.judgment_training import PreparedBundle, load_bundles, prepare_bundle, target_vector

STUDY = "contrast-scaling-v1"
KINDS = ("noul", "choice", "score")
ENDPOINTS = (0, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000)
MAX_INPUT_TOKENS = 1536
DEFAULT_MAX_UNITS = 8
DEFAULT_UNIT_BATCH_SIZE = 4
CONTRACT_CATEGORIES = (
    "claim_vs_completion", "permission_vs_execution", "unknown_vs_failure", "attribution_and_endorsement",
    "entity_binding", "action_binding", "temporal_scope", "reversal_and_current_state",
    "negation_and_quantifiers", "ordered_rubrics",
)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*parts):
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest(), 16)


def precision_policy(seed):
    return {
        "tf32": False, "seed": seed, "deterministic_algorithms": True,
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "cublas_workspace_config": ":4096:8",
    }


def apply_determinism(seed):
    """Apply the study's seed and precision policy before loading any model."""
    if type(seed) is not int or seed != 42:
        raise ValueError("contrast scaling v1 requires seed 42")
    # Set before model construction/CUDA BLAS work. Fixed seeds alone do not
    # make SDPA and grouped-convolution backward kernels reproducible.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return precision_policy(seed)


def _normalize_replay_pools(replay_pools):
    """Accept ``primitive -> source -> ids`` (or its source-first equivalent)."""
    if not isinstance(replay_pools, dict) or not replay_pools:
        raise ValueError("replay pools must be nonempty")
    if set(replay_pools) == set(KINDS):
        normalized = replay_pools
    else:
        normalized = {kind: {} for kind in KINDS}
        for source, by_kind in replay_pools.items():
            if not isinstance(by_kind, dict) or set(by_kind) != set(KINDS):
                raise ValueError("replay pools need every primitive for every source")
            for kind in KINDS:
                normalized[kind][source] = by_kind[kind]
    if set(normalized) != set(KINDS):
        raise ValueError("replay pools need Noul, Choice, and Score")
    ids = []
    result = {}
    for kind in KINDS:
        by_source = normalized[kind]
        if not isinstance(by_source, dict) or not by_source:
            raise ValueError(f"replay pool for {kind} is empty")
        result[kind] = {}
        for source, values in by_source.items():
            if not isinstance(source, str) or not source or not isinstance(values, list) or not values:
                raise ValueError("replay sources and IDs must be nonempty")
            if any(not isinstance(identity, str) or not identity for identity in values):
                raise ValueError("replay IDs must be nonempty strings")
            result[kind][source] = list(values)
            ids.extend(values)
    if len(ids) != len(set(ids)):
        raise ValueError("replay IDs must be globally unique")
    return result


def make_family_schedule(family_ids, replay_pools, total_steps, repeat_size=None, seed=42):
    """Build a deterministic ordered primary stream or a prefix-only repeat stream.

    Replay is a function of update position alone.  Thus a repeat stream built
    to 1000 updates uses the same old questions as the primary stream at every
    corresponding position, including positions 200--999 after restore.
    """
    if type(total_steps) is not int or total_steps < 1:
        raise ValueError("total_steps must be a positive integer")
    if not isinstance(family_ids, list) or not family_ids or any(not isinstance(x, str) or not x for x in family_ids):
        raise ValueError("family_ids must be nonempty strings")
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("family IDs must be unique")
    if repeat_size is None:
        if total_steps > len(family_ids):
            raise ValueError("ordered family IDs do not cover total_steps")
        active = family_ids
    else:
        if type(repeat_size) is not int or not 1 <= repeat_size <= len(family_ids):
            raise ValueError("repeat_size must be a valid family prefix")
        active = family_ids[:repeat_size]
    pools = _normalize_replay_pools(replay_pools)
    if set(family_ids) & {identity for by_source in pools.values() for ids in by_source.values() for identity in ids}:
        raise ValueError("new family and old replay IDs overlap")

    streams = {}
    for kind in KINDS:
        for source, values in sorted(pools[kind].items()):
            rng = random.Random(_stable_seed(STUDY, seed, kind, source))
            order = []
            while len(order) < total_steps:
                cycle = list(values)
                rng.shuffle(cycle)
                order.extend(cycle)
            streams[kind, source] = iter(order)
    schedule = []
    for position in range(total_steps):
        replay = []
        for kind in KINDS:
            sources = sorted(pools[kind])
            source = sources[position % len(sources)]
            replay.append({"origin": "old", "primitive": kind, "source": source, "id": next(streams[kind, source])})
        schedule.append({"step": position, "family_id": active[position % len(active)], "replay": replay})
    return schedule


def validate_family_coverage(records, expected_ids):
    """Hard-gate a prefix: every declared family member and relation endpoint exists."""
    if not isinstance(expected_ids, list) or not expected_ids or len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expected family IDs must be a nonempty unique list")
    by_id = {record.get("id"): record for record in records}
    if len(by_id) != len(records) or set(by_id) != set(expected_ids):
        raise ValueError("records do not exactly match the declared family prefix")
    for identity in expected_ids:
        family = by_id[identity]
        if family.get("group_id") != identity:
            raise ValueError(f"family {identity} must use its id as group_id")
        examples = family.get("examples")
        if not isinstance(examples, list) or len(examples) != 20:
            raise ValueError(f"family {identity} needs exactly 20 examples")
        members, by_case = set(), {}
        for index, example in enumerate(examples):
            case_id, question_id = example.get("case_id"), example.get("question_id")
            if not isinstance(case_id, str) or not case_id or not isinstance(question_id, str) or not question_id:
                raise ValueError(f"family {identity} example {index} needs case_id and question_id")
            if (case_id, question_id) in members:
                raise ValueError(f"family {identity} repeats a case/question member")
            members.add((case_id, question_id))
            by_case.setdefault(case_id, {})[question_id] = example
            question = example.get("question")
            kind = question.get("type") if isinstance(question, dict) else None
            expected_kind = {"n1": "noul", "n2": "noul", "n3": "noul", "c1": "choice", "s1": "score"}.get(question_id)
            if kind != expected_kind:
                raise ValueError(f"family {identity} example {index} has an unknown primitive")
            target, criteria = example.get("target"), question.get("criteria")
            if kind == "noul" and not (isinstance(target, dict) and set(target) == {"truth"} and type(target["truth"]) is bool):
                raise ValueError(f"family {identity} requires hard Noul targets")
            if kind == "choice":
                if not isinstance(criteria, dict) or not criteria or not (isinstance(target, dict) and set(target) == {"choice"}
                                                                          and type(target["choice"]) is str and target["choice"] in criteria):
                    raise ValueError(f"family {identity} requires hard Choice targets")
            if kind == "score":
                if not isinstance(criteria, (list, tuple)) or not criteria or not (
                    isinstance(target, dict) and set(target) == {"level_index"} and type(target["level_index"]) is int
                    and 0 <= target["level_index"] < len(criteria)
                ):
                    raise ValueError(f"family {identity} requires hard Score targets")
        expected_questions = {"n1", "n2", "n3", "c1", "s1"}
        if len(by_case) != 4 or any(set(case) != expected_questions for case in by_case.values()):
            raise ValueError(f"family {identity} needs exactly four cases with n1/n2/n3/c1/s1")
        for relation in family.get("relations", []):
            left, right = relation.get("left"), relation.get("right")
            if type(left) is not int or type(right) is not int or not (0 <= left < 20 and 0 <= right < 20) or left == right:
                raise ValueError(f"family {identity} has an invalid relation endpoint")
            first, second = examples[left], examples[right]
            first_question, second_question = first["question"], second["question"]
            if first_question["type"] != second_question["type"]:
                raise ValueError(f"family {identity} relation has incompatible answer spaces")
            kind = relation.get("kind")
            if kind == "complement":
                if first_question["type"] != "noul" or first["target"]["truth"] == second["target"]["truth"]:
                    raise ValueError(f"family {identity} complement relation has incompatible targets")
            elif kind == "invariant":
                if first_question["criteria"] != second_question["criteria"] or first["target"] != second["target"]:
                    raise ValueError(f"family {identity} invariant relation has incompatible targets")
            elif kind == "flip":
                if first_question["criteria"] != second_question["criteria"] or first["target"] == second["target"]:
                    raise ValueError(f"family {identity} flip relation has incompatible targets")
            else:
                raise ValueError(f"family {identity} relation kind is unsupported")
    return by_id


def ordered_family_ids(manifest):
    """Read the one canonical ordering and prove the three nested prefixes."""
    ordered = manifest.get("ordered_family_ids", manifest.get("family_ids"))
    if not isinstance(ordered, list) or not ordered or any(not isinstance(value, str) or not value for value in ordered):
        raise ValueError("manifest needs ordered_family_ids")
    nested = manifest.get("nested_prefixes", manifest.get("prefixes"))
    if not isinstance(nested, dict):
        raise ValueError("manifest needs nested_prefixes")
    for size in (200, 1000, 5000):
        declared = nested.get(str(size), nested.get(size))
        if isinstance(declared, dict):
            declared = declared.get("family_ids")
        if declared != ordered[:size]:
            raise ValueError(f"manifest prefix {size} is not the ordered family prefix")
    return ordered


def validate_training_contract(records, manifest):
    """Validate the fixed 5000-family train population before model preparation."""
    ordered = ordered_family_ids(manifest)
    if len(ordered) != 5000 or len(records) != 5000:
        raise ValueError("contrast scaling requires exactly 5000 training families")
    by_id = validate_family_coverage(records, ordered)
    categories = []
    for identity in ordered:
        provenance = by_id[identity].get("provenance", {})
        if provenance.get("assigned_split") != "train":
            raise ValueError(f"family {identity} is not assigned to train")
        category = provenance.get("category")
        if not isinstance(category, str) or not category:
            raise ValueError(f"family {identity} has no category")
        categories.append(category)
    validate_prefix_category_balance(categories, CONTRACT_CATEGORIES)
    return ordered


def validate_prefix_category_balance(categories, allowed_categories):
    """Require exact category balance in every declared nested training prefix."""
    if not isinstance(categories, list) or len(categories) != 5000 or set(categories) != set(allowed_categories):
        raise ValueError("training families must use exactly the ten contract categories")
    for size, expected in ((200, 20), (1000, 100), (5000, 500)):
        counts = {category: categories[:size].count(category) for category in allowed_categories}
        if set(counts.values()) != {expected}:
            raise ValueError(f"prefix {size} must contain {expected} families per category")


def _group_loss(logits, group):
    target = target_vector(group)
    if group.primitive == "noul":
        return F.binary_cross_entropy_with_logits(logits[0], logits.new_tensor(target[1]))
    return -(logits.new_tensor(target) * logits.log_softmax(-1)).sum()


def _complete_microbatches(bundle, max_units, *, selected=None):
    """Yield prompt slices that pack whole candidate groups without crossing the cap."""
    if type(max_units) is not int or max_units < 1:
        raise ValueError("max_units must be a positive integer")
    selected_ids = None if selected is None else {id(group) for group in selected}
    prompts, kinds, members = [], [], []
    for group in bundle.groups:
        if selected_ids is not None and id(group) not in selected_ids:
            continue
        indices = list(group.indices)
        if not indices or max(indices) >= len(bundle.prompts) or min(indices) < 0:
            raise ValueError("group indices are outside its prepared bundle")
        if len(indices) > max_units:
            raise ValueError("one complete candidate group exceeds max_units")
        if prompts and len(prompts) + len(indices) > max_units:
            yield prompts, kinds, members
            prompts, kinds, members = [], [], []
        start = len(prompts)
        prompts.extend(bundle.prompts[index] for index in indices)
        kinds.extend(bundle.kinds[index] for index in indices)
        members.append((group, tuple(range(start, start + len(indices)))) )
    if prompts:
        yield prompts, kinds, members


def _component_groups(family, replay):
    if not isinstance(family, PreparedBundle):
        raise ValueError("new training item must be a prepared family bundle")
    if set(replay) != set(KINDS):
        raise ValueError("one old replay bundle per primitive is required")
    result = {("new", kind): [] for kind in KINDS}
    for group in family.groups:
        if group.primitive not in KINDS:
            raise ValueError("new family has an unknown primitive")
        result["new", group.primitive].append((family, group))
    for kind in KINDS:
        bundle = replay[kind]
        if not isinstance(bundle, PreparedBundle) or len(bundle.groups) != 1 or bundle.groups[0].primitive != kind:
            raise ValueError("old replay must contain exactly one matching primitive question")
        result["old", kind] = [(bundle, bundle.groups[0])]
    if any(not values for values in result.values()):
        raise ValueError("all six origin/primitive components need supervision")
    return result


def _score_bundle_groups(model, bundle, selected, *, max_units, unit_batch_size):
    wanted = {id(group) for group in selected}
    by_group = {}
    for prompts, kinds, members in _complete_microbatches(bundle, max_units, selected=selected):
        logits = model.score_prompts(prompts, kinds, unit_batch_size=unit_batch_size)
        for group, local in members:
            by_group[id(group)] = _group_loss(logits[list(local)], group)
    if set(by_group) != wanted:
        raise RuntimeError("candidate microbatch did not score every requested group")
    return [by_group[id(group)] for group in selected]


def family_loss_components(model, family, replay, *, max_units=DEFAULT_MAX_UNITS, unit_batch_size=DEFAULT_UNIT_BATCH_SIZE):
    """Return the six equally weighted primitive means for one complete update.

    This pure differentiable form supports reference checks.  The training loop
    below uses the same weights but backpropagates each complete microbatch as
    soon as it is scored, bounding activation lifetime.
    """
    components = _component_groups(family, replay)
    grouped = {}
    for key, values in components.items():
        bundle = values[0][0]
        grouped[key] = _score_bundle_groups(
            model, bundle, [group for _, group in values], max_units=max_units, unit_batch_size=unit_batch_size
        )
    means = {f"{origin}/{kind}": torch.stack(values).mean() for (origin, kind), values in grouped.items()}
    total = sum(means.values()) / 6
    return {"total": total, "components": means, "counts": {key: len(value) for key, value in ((f"{o}/{k}", v) for (o, k), v in grouped.items())}}


def backward_family_loss(model, family, replay, *, max_units=DEFAULT_MAX_UNITS, unit_batch_size=DEFAULT_UNIT_BATCH_SIZE):
    """Backpropagate a family update in bounded complete candidate microbatches."""
    components = _component_groups(family, replay)
    totals = {}
    for (origin, kind), values in components.items():
        bundle = values[0][0]
        wanted = [group for _, group in values]
        count = len(wanted)
        numeric = []
        for prompts, kinds, chosen in _complete_microbatches(bundle, max_units, selected=wanted):
            logits = model.score_prompts(prompts, kinds, unit_batch_size=unit_batch_size)
            losses = [_group_loss(logits[list(local)], group) for group, local in chosen]
            micro = torch.stack(losses).sum() / (6 * count)
            if not torch.isfinite(micro):
                raise RuntimeError("non-finite complete-candidate microbatch loss")
            micro.backward()
            numeric.extend(loss.detach().item() for loss in losses)
        if len(numeric) != count:
            raise RuntimeError("missing complete candidate group during backward")
        totals[f"{origin}/{kind}"] = sum(numeric) / count
    return {"loss": sum(totals.values()) / 6, "components": totals, "counts": {key: len(value) for key, value in ((f"{o}/{k}", v) for (o, k), v in components.items())}}


def save_resume_state(path, model, optimizer, state):
    """Small-model checkpoint helper used by the CPU resume proof."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state,
         "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        path / "training.pt",
    )


def restore_resume_state(path, model, optimizer):
    payload = torch.load(Path(path) / "training.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng"):
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]


def restore_training_state(path, optimizer):
    """Restore the training.pt written by JudgmentModel.save_checkpoint."""
    payload = torch.load(Path(path) / "training.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng"):
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]


def save_checkpoint(engine, path, optimizer, *, step, state, stream):
    metadata = {"study": STUDY, "step": step, "stream": stream, "max_units": state["max_units"]}
    identity = engine.model.save_checkpoint(path, metadata=metadata, optimizer=optimizer, training_state=state)
    engine.model_id = identity
    return identity


def _max_difference(left, right):
    if len(left) != len(right):
        return float("inf")
    return max((a.detach() - b.detach()).abs().max().item() for a, b in zip(left, right, strict=True)) if left else 0.0


@torch.inference_mode()
def _model_probabilities(model, bundle, unit_batch_size):
    logits = model.score_prompts(bundle.prompts, bundle.kinds, unit_batch_size=unit_batch_size)
    values = []
    for group in bundle.groups:
        scores = logits[list(group.indices)]
        values.extend(([scores[0].sigmoid(), 1 - scores[0].sigmoid()] if group.primitive == "noul" else scores.softmax(-1)))
    return [value.detach().float().cpu().clone() for value in values]


def production_resume_equivalence(engine, optimizer, checkpoint, load_engine, family, replay, *, max_units, unit_batch_size,
                                  tolerance=1e-7):
    """Prove production checkpoint reload and its next bounded update are equivalent.

    ``load_engine`` must load the saved production checkpoint, not clone the
    in-memory model.  This is intentionally used only in smoke, after the
    checkpoint has been committed to disk.
    """
    before_frozen = frozen_digest(engine.model)
    original_probabilities = _model_probabilities(engine.model, family, unit_batch_size)
    optimizer.zero_grad(set_to_none=True)
    backward_family_loss(engine.model, family, replay, max_units=max_units, unit_batch_size=unit_batch_size)
    torch.nn.utils.clip_grad_norm_([p for p in engine.model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    optimizer.step()
    original_trainable = [p.detach().float().cpu().clone() for p in engine.model.parameters() if p.requires_grad]
    after_frozen = frozen_digest(engine.model)
    if before_frozen != after_frozen:
        raise RuntimeError("frozen body changed during resume proof")
    # The uninterrupted branch is no longer needed.  Move it off-device before
    # constructing the saved branch so a 16 GB smoke host holds one 2B model.
    engine.model.to("cpu")
    optimizer.zero_grad(set_to_none=True)
    optimizer.state.clear()
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    reloaded = load_engine(checkpoint)
    reloaded_optimizer = optimizer_for(reloaded.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    restored_state = restore_training_state(checkpoint, reloaded_optimizer)
    if before_frozen != frozen_digest(reloaded.model):
        raise RuntimeError("saved/reloaded frozen body differs")
    restored_probabilities = _model_probabilities(reloaded.model, family, unit_batch_size)
    probability_difference = _max_difference(original_probabilities, restored_probabilities)
    if probability_difference > tolerance:
        raise RuntimeError("saved/reloaded probabilities differ")
    reloaded_optimizer.zero_grad(set_to_none=True)
    backward_family_loss(reloaded.model, family, replay, max_units=max_units, unit_batch_size=unit_batch_size)
    torch.nn.utils.clip_grad_norm_([p for p in reloaded.model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
    reloaded_optimizer.step()
    restored_trainable = [p.detach().float().cpu() for p in reloaded.model.parameters() if p.requires_grad]
    trainable_difference = _max_difference(original_trainable, restored_trainable)
    if after_frozen != frozen_digest(reloaded.model):
        raise RuntimeError("frozen body changed during resume proof")
    if trainable_difference > tolerance:
        raise RuntimeError("saved/reloaded next update differs")
    return {"passed": True, "restored_state": restored_state, "max_probability_difference": probability_difference,
            "max_logit_difference": probability_difference, "max_trainable_difference": trainable_difference,
            "frozen_before": before_frozen, "frozen_after": after_frozen, "tolerance": tolerance,
            "probability_probe": {"requires_grad": False, "device": "cpu"}}


def _prepare_family(engine, record):
    prepared = prepare_bundle(engine, record)
    longest = max(map(len, prepared.prompts))
    if longest > MAX_INPUT_TOKENS:
        raise ValueError(f"family {record['id']} exceeds {MAX_INPUT_TOKENS} tokens; no dropping allowed")
    return prepared


def prepare_families(engine, records, expected_ids):
    """Preflight all family prompts.  An oversized member aborts the whole run."""
    validate_family_coverage(records, expected_ids)
    prepared = {}
    for record in records:
        prepared[record["id"]] = _prepare_family(engine, record)
    return prepared


def validate_prepared_group_widths(prepared, *, max_units):
    """Reject a unit budget before any smoke subset can hide a wider candidate."""
    if type(max_units) is not int or max_units < 1:
        raise ValueError("max_units must be a positive integer")
    required = 0
    for identity, bundle in prepared.items():
        if not isinstance(bundle, PreparedBundle):
            raise ValueError(f"prepared item {identity} is not a bundle")
        for group in bundle.groups:
            width = len(group.indices)
            required = max(required, width)
            if width > max_units:
                raise ValueError(f"prepared {identity} requires {width} units; max_units is {max_units}")
    if not required:
        raise ValueError("prepared population has no candidate groups")
    return required


def _single_question(identity, example, source):
    return {"id": identity, "examples": [example], "relations": [], "provenance": {"dataset": source}}


def prepare_canonical_replay(engine, path=Path("data/processed-v0.2/train.jsonl")):
    """Prepare the pre-existing canonical 800-per-source replay population.

    The source/hash ordering intentionally matches the preceding paired runner.
    Replay examples are selected before any new-family ordering is considered.
    """
    selected, prepared, pools = {}, {}, {kind: {} for kind in KINDS}
    for record in sorted(load_bundles(path), key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()):
        provenance = record.get("provenance", {})
        if provenance.get("assigned_split") != "train":
            raise ValueError("original replay contains a nontraining record")
        source = provenance.get("dataset")
        if not isinstance(source, str) or not source:
            raise ValueError("original replay record has no source")
        if len(selected.get(source, [])) >= 800:
            continue
        example = record.get("examples", [None])[0]
        question = example.get("question") if isinstance(example, dict) else None
        kind = question.get("type") if isinstance(question, dict) else None
        if kind not in KINDS:
            raise ValueError("original replay has an unknown primitive")
        identity = "old/" + record["id"]
        bundle = prepare_bundle(engine, _single_question(identity, example, source))
        if max(map(len, bundle.prompts)) > MAX_INPUT_TOKENS:
            raise ValueError(f"canonical replay {identity} exceeds {MAX_INPUT_TOKENS} tokens")
        selected.setdefault(source, []).append(identity)
        prepared[identity] = bundle
        pools[kind].setdefault(source, []).append(identity)
    if not selected or any(len(rows) != 800 for rows in selected.values()):
        raise ValueError("need exactly 800 canonical replay records per original source")
    if any(not pools[kind] for kind in KINDS):
        raise ValueError("canonical replay is missing a primitive")
    return prepared, pools


def run_stream(engine, schedule, prepared, *, output, max_seconds, max_units=DEFAULT_MAX_UNITS, start_step=0,
               optimizer=None, stream="primary", progress=None):
    """Train a contiguous schedule range and durably save a resumable partial state."""
    if max_seconds <= 0:
        raise ValueError("max_seconds must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = engine.model
    optimizer = optimizer or optimizer_for(model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    frozen_before = frozen_digest(model)
    started, history = time.monotonic(), []
    gradient_check = None
    attempted_step, update_phase = None, "boundary"
    try:
        if start_step == 0:
            state = {"step": 0, "next_step": 0, "max_units": max_units, "stream": stream}
            save_checkpoint(engine, output / "step-0000", optimizer, step=0, state=state, stream=stream)
        for row in schedule[start_step:]:
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("training time cap reached")
            attempted_step, update_phase = row["step"] + 1, "pre-step"
            family = prepared[row["family_id"]]
            replay = {item["primitive"]: prepared[item["id"]] for item in row["replay"]}
            optimizer.zero_grad(set_to_none=True)
            model.train()
            result = backward_family_loss(model, family, replay, max_units=max_units, unit_batch_size=engine.unit_batch_size)
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise RuntimeError("non-finite gradient norm")
            if gradient_check is None:
                gradient_check = {
                    "adapters": any(p.grad is not None and p.grad.abs().max().item() > 0
                                    for name, p in model.named_parameters() if ".lora_" in name),
                    "heads": any(p.grad is not None and p.grad.abs().max().item() > 0
                                 for name, p in model.named_parameters() if name.startswith(("binary.", "compatibility."))),
                    "frozen": all(p.grad is None for p in model.parameters() if not p.requires_grad),
                    "finite_norm": True,
                }
                if not all(gradient_check.values()):
                    raise RuntimeError("adapter/head/frozen gradient smoke check failed")
            update_phase = "optimizer-step"
            optimizer.step()
            update_phase = "post-step"
            engine.synchronize()
            entry = {"step": row["step"] + 1, "loss": result["loss"], "by_component": result["components"],
                     "gradient_norm": norm.item(), "training_seconds": time.monotonic() - started}
            history.append(entry)
            update_phase = "boundary"
            with (output / "history.jsonl").open("a") as stream_file:
                stream_file.write(json.dumps(entry, allow_nan=False) + "\n")
            if entry["step"] % 20 == 0:
                print(f"{stream} update {entry['step']}/{len(schedule)} loss {entry['loss']:.4f}", flush=True)
            if progress is not None:
                progress(stream, entry["step"], entry)
            if entry["step"] in ENDPOINTS:
                state = {"step": entry["step"], "next_step": entry["step"], "max_units": max_units, "stream": stream}
                save_checkpoint(engine, output / f"step-{entry['step']:04d}", optimizer, step=entry["step"], state=state, stream=stream)
        frozen_after = frozen_digest(model)
        if frozen_before != frozen_after:
            raise RuntimeError("frozen body weights changed")
        result = {"status": "complete", "completed_updates": start_step + len(history), "history_updates": len(history),
                  "training_seconds": time.monotonic() - started, "frozen_before": frozen_before, "frozen_after": frozen_after,
                  "max_units": max_units, "stream": stream, "gradient_check": gradient_check}
        write(output / "training.json", result)
        if progress is not None:
            progress(stream, result["completed_updates"], result)
        return result
    except BaseException as error:
        completed = start_step + len(history)
        resumable = update_phase == "boundary"
        state = {
            "step": completed, "next_step": completed if resumable else None, "completed_updates": completed,
            "attempted_step": attempted_step if attempted_step is not None else completed, "resumable": resumable,
            "max_units": max_units, "stream": stream, "status": "partial",
        }
        try:
            save_checkpoint(engine, output / f"partial-{completed:04d}", optimizer, step=completed, state=state, stream=stream)
        except Exception:
            pass
        write(output / "training.json", {"status": "partial", "completed_updates": completed,
                                           "attempted_step": state["attempted_step"], "resumable": resumable,
                                           "training_seconds": time.monotonic() - started, "error": repr(error), "stream": stream})
        if progress is not None:
            progress(stream, completed, {"status": "partial", "error": repr(error)})
        raise


def _checkpoint_fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256(str(path).encode())
    for file in sorted(path.rglob("*")):
        if file.is_file():
            digest.update(str(file.relative_to(path)).encode())
            digest.update(file.read_bytes())
    if not any(path.rglob("*")):
        raise ValueError(f"starting checkpoint is empty: {path}")
    return digest.hexdigest()


def prepared_prompt_fingerprint(prepared):
    digest = hashlib.sha256()
    for identity, bundle in sorted(prepared.items()):
        digest.update(identity.encode())
        digest.update(json.dumps({"prompts": bundle.prompts, "kinds": bundle.kinds}, separators=(",", ":")).encode())
    return digest.hexdigest()


def _package_versions():
    result = {}
    for name in ("torch", "transformers", "peft", "fla-core"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unavailable"
    return result


def fingerprint(args, *, prompt_fingerprint=None):
    files = [
        Path(__file__), Path("src/openjev/judgment_model.py"), Path("src/openjev/judgment_training.py"),
        Path("src/openjev/judgment_cli.py"), Path("src/openjev/judgments.py"), Path("src/openjev/prompts.py"),
        Path("src/openjev/runtime.py"), Path("experiments/judgment_pipeline.py"),
    ]
    manifest = _manifest_path(args.data)
    return {"study": STUDY, "source_sha256": {str(path): sha(path) for path in files},
            "train_sha256": sha(args.data / "train.jsonl"), "manifest_sha256": sha(manifest) if manifest else None,
            "replay_sha256": sha("data/processed-v0.2/train.jsonl"), "backend": "fla", "max_input_tokens": MAX_INPUT_TOKENS,
            "unit_batch_size": args.unit_batch_size, "max_units": args.max_units, "seed": args.seed,
            "precision": precision_policy(args.seed), "starting_checkpoint": _checkpoint_fingerprint(args.checkpoint),
            "prepared_prompt_fingerprint": prompt_fingerprint, "packages": _package_versions()}


def validate_smoke_gate(gate, expected_fingerprint):
    if not (gate.get("mode") == "smoke" and gate.get("status") == "complete" and gate.get("passed")
            and gate.get("completed_requested_pass") and gate.get("fingerprint") == expected_fingerprint):
        raise ValueError("full training requires a completed matching smoke fingerprint")


def _manifest_path(data):
    candidates = (data / "train" / "manifest.json", data / "manifest.json")
    return next((path for path in candidates if path.exists()), None)


def _new_engine(checkpoint, unit_batch_size):
    return load_judgment_engine(
        checkpoint=checkpoint, trainable=True, backend="fla", max_input_tokens=MAX_INPUT_TOKENS,
        unit_batch_size=unit_batch_size,
    )


def run(args):
    """Write durable task status around all preparation and training work."""
    args.output = Path(args.output)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"output already exists: {args.output}") from error
    status = {"mode": args.mode, "status": "preparing", "completed_updates": {}, "started_unix": time.time()}
    write(args.output / "report.json", status)
    try:
        return _run(args, status)
    except BaseException as error:
        status.update({"status": "failed", "error": repr(error), "finished_unix": time.time()})
        write(args.output / "report.json", status)
        raise


def _run(args, run_status):
    """Perform preflight, smoke, or the primary/repeat streams without touching test outcomes."""
    manifest_path = _manifest_path(args.data)
    if manifest_path is None:
        raise FileNotFoundError("expected train/manifest.json or manifest.json")
    records = load_bundles(args.data / "train.jsonl")
    ordered = validate_training_contract(records, json.loads(manifest_path.read_text()))
    precision = apply_determinism(args.seed)
    engine = _new_engine(args.checkpoint, args.unit_batch_size)
    prepared = prepare_families(engine, records, ordered)
    old_prepared, replay_pools = prepare_canonical_replay(engine)
    prepared.update(old_prepared)
    required_max_units = validate_prepared_group_widths(prepared, max_units=args.max_units)
    report = {
        **run_status,
        "mode": args.mode,
        "status": "preflight-complete",
        "fingerprint": fingerprint(args, prompt_fingerprint=prepared_prompt_fingerprint(prepared)),
        "precision": precision,
        "families": len(ordered),
        "prepared_family_members": sum(len(bundle.groups) for identity, bundle in prepared.items() if identity in set(ordered)),
        "replay_by_primitive_source": {kind: {source: len(values) for source, values in by_source.items()}
                                       for kind, by_source in replay_pools.items()},
        "max_input_tokens": MAX_INPUT_TOKENS,
        "required_max_units": required_max_units,
        "no_dropped_new_examples": True,
    }

    def progress(stream, completed, detail):
        completed_updates = dict(report.get("completed_updates", {}))
        completed_updates[stream] = completed
        report.update({"status": "running", "active_stream": stream, "completed_updates": completed_updates,
                       "last_progress": detail, "updated_unix": time.time()})
        run_status.update(report)
        write(args.output / "report.json", report)

    if args.mode == "preflight":
        write(args.output / "report.json", report)
        return report
    if args.mode == "smoke":
        schedule = make_family_schedule(ordered, replay_pools, total_steps=args.smoke_steps + 1, seed=args.seed)
        optimizer = optimizer_for(engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
        training = run_stream(engine, schedule[:args.smoke_steps], prepared, output=args.output / "smoke", max_seconds=args.max_seconds,
                              max_units=args.max_units, optimizer=optimizer, stream="smoke", progress=progress)
        final = args.output / "smoke" / "final"
        state = {"step": args.smoke_steps, "next_step": args.smoke_steps, "max_units": args.max_units, "stream": "smoke"}
        save_checkpoint(engine, final, optimizer, step=args.smoke_steps, state=state, stream="smoke")
        def load_saved(path):
            return _new_engine(path, args.unit_batch_size)

        smoke_row = schedule[args.smoke_steps]
        smoke_replay = {item["primitive"]: prepared[item["id"]] for item in smoke_row["replay"]}
        resume_proof = production_resume_equivalence(
            engine, optimizer, final, load_saved, prepared[smoke_row["family_id"]], smoke_replay,
            max_units=args.max_units, unit_batch_size=args.unit_batch_size,
        )
        report.update({"mode": "smoke", "status": "complete", "passed": training["status"] == "complete" and resume_proof["passed"],
                       "completed_requested_pass": training["completed_updates"] == args.smoke_steps,
                       "reload_resume_state": resume_proof["restored_state"], "resume_equivalence": resume_proof,
                       "training": training})
        write(args.output / "report.json", report)
        return report
    if args.max_seconds <= 0:
        raise ValueError("full mode requires a positive training budget")
    if args.smoke_gate is None:
        raise ValueError("full mode requires --smoke-gate")
    validate_smoke_gate(json.loads(args.smoke_gate.read_text()), report["fingerprint"])
    primary = make_family_schedule(ordered, replay_pools, total_steps=5000, seed=args.seed)
    primary_result = run_stream(engine, primary, prepared, output=args.output / "primary", max_seconds=args.max_seconds,
                                max_units=args.max_units, stream="primary", progress=progress)
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    repeat_engine = _new_engine(args.output / "primary" / "step-0200", args.unit_batch_size)
    repeat_optimizer = optimizer_for(repeat_engine.model, SimpleNamespace(lr=5e-5, head_lr=2.5e-5))
    restored = restore_training_state(args.output / "primary" / "step-0200", repeat_optimizer)
    if restored.get("next_step") != 200:
        raise ValueError("the repeat branch must restore the primary 200-family checkpoint")
    repeat = make_family_schedule(ordered, replay_pools, total_steps=1000, repeat_size=200, seed=args.seed)
    repeat_result = run_stream(repeat_engine, repeat, prepared, output=args.output / "repeat-200", max_seconds=args.max_seconds,
                               max_units=args.max_units, start_step=200, optimizer=repeat_optimizer, stream="repeat-200",
                               progress=progress)
    report.update({"status": "complete", "primary": primary_result, "repeat_200": repeat_result,
                   "repeat_restored_state": restored, "test_outcomes_read": False})
    write(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/contrast-scaling-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/judgment-full-v0.2/final"))
    parser.add_argument("--mode", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--max-seconds", type=float, default=12 * 60 * 60)
    parser.add_argument("--max-units", type=int, default=DEFAULT_MAX_UNITS)
    parser.add_argument("--unit-batch-size", type=int, default=DEFAULT_UNIT_BATCH_SIZE)
    parser.add_argument("--smoke-steps", type=int, default=1)
    parser.add_argument("--smoke-gate", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_units < 1 or args.unit_batch_size < 1 or args.unit_batch_size > args.max_units:
        raise ValueError("unit batch size must be positive and no larger than max units")
    if args.smoke_steps < 1:
        raise ValueError("smoke_steps must be positive")
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
