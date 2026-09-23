"""Locked, exploratory development evaluation for calibrated-screen-v1.

Reuses the preserved factorial prediction/calibration stages without editing their
sources. All selected endpoints precede all fits; all fits precede development
reads. The original TEST/Jev quarantine is never an input.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from experiments import calibrated_training as training
from experiments import contrast_factorial_pipeline as pipeline
from experiments import contrast_factorial_training as factorial
from experiments.calibrated_integrity import load_verified_targets
from experiments.contrast_factorial_reporting import _percentile, summarize_factorial

STUDY = training.STUDY
ARMS = training.ARMS
REPO_ROOT = training.REPO_ROOT
SCOPE = "EXPLORATORY — previously inspected development assessment and fixed legacy regressions"
VARIANTS = ("raw", "global", "per_primitive")
COMPARISONS = (("H1", "H0"), ("Q0", "H0"), ("J0", "H0"), ("Q1", "Q0"),
               ("J1", "J0"), ("Q1", "H1"), ("J1", "H1"), ("J0", "Q0"), ("J1", "Q1"))
_CONTEXT_LOCK = threading.Lock()
EVALUATION_SOURCES = (
    "experiments/calibrated_evaluation.py", "experiments/contrast_factorial_pipeline.py",
    "experiments/contrast_factorial_reporting.py", "experiments/revised_metrics.py",
    "experiments/calibrated_integrity.py", "experiments/calibrated_qwen_teacher.py",
    "experiments/jev_archive.py", "experiments/jev_replay.py",
    "experiments/revised_evaluation.py", "experiments/temperature_diagnostic.py",
)


def _models(arms):
    if (not isinstance(arms, (list, tuple)) or not arms or len(set(arms)) != len(arms)
            or not set(arms) <= {"BASE", *ARMS} or not set(arms) & set(ARMS)):
        raise ValueError("explicit unique selected arms must contain a trained arm, with optional BASE")
    return tuple(name for name in ("BASE", *ARMS) if name in arms)


@contextmanager
def runner_context(models):
    """Restore every adapted constant even if a stage fails; reject overlap."""
    if not _CONTEXT_LOCK.acquire(blocking=False):
        raise RuntimeError("calibrated evaluation context already active")
    before = pipeline.STUDY, pipeline.MODELS, pipeline.ARMS
    try:
        pipeline.STUDY, pipeline.MODELS, pipeline.ARMS = STUDY, tuple(models), tuple(m for m in models if m != "BASE")
        yield
    finally:
        pipeline.STUDY, pipeline.MODELS, pipeline.ARMS = before
        _CONTEXT_LOCK.release()


def _file_spec(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": pipeline.sha256_file(path)}


def _check_spec(spec, label):
    path = Path(spec["path"])
    if not path.is_file() or pipeline.sha256_file(path) != spec["sha256"]:
        raise ValueError(f"{label} changed or missing")
    return path


def _source_mapping(repo, mapping, label, required=()):
    if not isinstance(mapping, dict) or not mapping or not set(required) <= set(mapping):
        raise ValueError(f"{label} source inventory is incomplete")
    return {str(pipeline._verify_file(repo, path, digest, label)): digest for path, digest in mapping.items()}


def _verify_screen_files(endpoint):
    for label, spec in endpoint["input_pins"].items():
        _check_spec(spec, label)
    for path, digest in endpoint["source_pins"].items():
        _check_spec({"path": path, "sha256": digest}, "source")
    for spec in endpoint["targets"].values():
        _check_spec(spec, "teacher targets")
        # The target file also binds every retained raw observation by SHA.
        load_verified_targets(spec["path"], endpoint["training_suite"])
    pipeline._verify_endpoint_files(endpoint)
    approval = pipeline._read_json(endpoint["approval"]["path"])
    base = pipeline._repo_path(endpoint["repo_root"], approval["base_checkpoint"], "base checkpoint")
    if pipeline.tree_sha256(base) != approval["base_checkpoint_sha256"]:
        raise ValueError("approved base checkpoint changed after locking")


def _validate_rng_payload(payload, common):
    """Check restorability without touching global CPU RNG or opening CUDA.

    This screen's pinned CUDA runtime serializes Philox as two uint64 values
    (seed and offset), totaling 16 bytes. This is a screen-specific format gate,
    not a promise about every PyTorch version or accelerator generator.
    """
    import torch

    def byte_vector(value, length=None):
        return (isinstance(value, torch.Tensor) and value.device.type == "cpu"
                and value.dtype == torch.uint8 and value.layout == torch.strided
                and value.ndim == 1 and value.is_contiguous()
                and (length is None or value.numel() == length))

    cpu_rng = payload.get("torch_rng")
    if not byte_vector(cpu_rng):
        raise ValueError("endpoint CPU RNG tensor layout is invalid")
    try:
        torch.Generator(device="cpu").set_state(cpu_rng)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("endpoint CPU RNG state is not restorable") from error
    cuda_rng = payload.get("cuda_rng")
    declared = common.get("cuda_rng_sha256")
    if not isinstance(cuda_rng, list) or not isinstance(declared, list) or len(cuda_rng) != len(declared):
        raise ValueError("endpoint CUDA RNG count differs from common start")
    if any(not byte_vector(value, 16) for value in cuda_rng):
        raise ValueError("endpoint CUDA RNG layout differs from the pinned 16-byte screen format")


def _validate_arm(repo, arm, checkpoint, manifest, manifest_sha, approval, common, targets):
    # Both original consumers read their own STUDY globals while validating.
    adapted_manifest = {**manifest, "arms": {name: manifest["arms"]["B"] for name in ARMS}}
    with runner_context((arm,)), training.runner_context(arm):
        endpoint = pipeline._validate_arm_endpoint(checkpoint, arm=arm, manifest=adapted_manifest,
            manifest_sha=manifest_sha, approval={**approval, "_repo_root": str(repo)}, coordinator_start=common)
        fp = endpoint["fingerprint"]
        payload = factorial.validate_resume_checkpoint(checkpoint, arm=arm, fingerprint=fp, output=checkpoint.parent)
    _validate_rng_payload(payload, common)
    expected = {"loss": training.loss_config(arm), "training_source_arm": "B", "updates": 400,
                "adapter_lr": 5e-5, "head_lr": 2.5e-5, "gradient_clip_norm": 1.0}
    if any(fp.get(key) != value for key, value in expected.items()):
        raise ValueError(f"arm {arm} loss or calibrated training recipe mismatch")
    target = targets.get(arm[0])
    if fp.get("targets_sha256") != (target["sha256"] if target else None):
        raise ValueError(f"arm {arm} target fingerprint mismatch")
    required_sources = ("experiments/calibrated_training.py", "experiments/calibrated_targets.py",
                        "experiments/calibrated_integrity.py", "reports/calibrated-screen-v1/PROTOCOL.md")
    sources = _source_mapping(repo, fp.get("calibrated_source_sha256"), "calibrated", required_sources)
    sources.update(_source_mapping(repo, fp.get("production_source_sha256"), "production", factorial.PRODUCTION_SOURCES))
    for name in ("source_fingerprints", "data_fingerprint"):
        sources.update(_source_mapping(repo, fp.get(name), name))
    replay = pipeline._verify_file(repo, fp.get("replay_path", ""), fp.get("replay_sha256"), "replay")
    sources[str(replay)] = fp["replay_sha256"]
    initial = {"state": {}, "param_groups": payload["optimizer"]["param_groups"]}
    if common["optimizer_initial_sha256"] != pipeline.canonical_hash(initial):
        raise ValueError(f"arm {arm} fresh Adam recipe differs from common start")
    state = payload["state"]
    progress_path, report_path = checkpoint.parent / "progress.json", checkpoint.parent / "report.json"
    progress, report = pipeline._read_json(progress_path), pipeline._read_json(report_path)
    for key in ("study", "arm", "status", "completed_updates", "next_position", "history_sha256",
                "frozen_before", "frozen_after", "frozen_verified_at", "complete_boundary", "resumable", "update_phase"):
        if progress.get(key) != state.get(key):
            raise ValueError(f"arm {arm} progress differs from saved endpoint")
    reported = report.get("training", {})
    transient = {"checkpoint", "session_time_seconds"}
    reported_core = {key: value for key, value in reported.items() if key in state or key not in transient}
    session_time = reported.get("session_time_seconds", 0.0)
    if type(session_time) not in (int, float) or not math.isfinite(session_time) or session_time < 0:
        raise ValueError(f"arm {arm} report session time is invalid")
    if (report.get("study") != STUDY or report.get("arm") != arm or report.get("mode") != "train"
            or report.get("status") != "complete" or report.get("passed") is not True
            or report.get("completed_requested_pass") is not True or report.get("fingerprint") != fp
            or reported_core != state or reported.get("checkpoint") != str(checkpoint)):
        raise ValueError(f"arm {arm} completed training report differs from saved endpoint")
    pins = {f"{arm}/{path.name}": _file_spec(path) for path in
            (progress_path, report_path, checkpoint.parent / "history.jsonl")}
    return endpoint, sources, pins


def create_endpoint_lock(*, repo_root, manifest_path, training_root, output_root, arms,
                         protocol_path=None, preserved_path=None, prior_endpoint_lock=None,
                         common_start_path=None, targets=None):
    """Validate exactly the explicit completed population; never discover winners."""
    repo = Path(repo_root).resolve()
    models = _models(arms)
    screen = repo / "reports" / STUDY
    output = Path(output_root).resolve()
    run = Path(training_root).resolve()
    if screen not in output.parents:
        raise ValueError("evaluation output must be a new directory under reports/calibrated-screen-v1")
    protocol = Path(protocol_path or screen / "PROTOCOL.md").resolve()
    preserved_path = Path(preserved_path or screen / "preserved-artifacts.json").resolve()
    prior_path = Path(prior_endpoint_lock or repo / "reports/contrast-factorial-v1/pipeline-v1/endpoint-lock.json").resolve()
    common_path = Path(common_start_path or screen / "common-start.json").resolve()
    manifest_path = Path(manifest_path).resolve()
    targets = {key: Path(value).resolve() for key, value in (targets or {}).items()}
    needed = {arm[0] for arm in models if arm[0] in {"J", "Q"}}
    if set(targets) != needed:
        raise ValueError("target paths must name exactly the selected Q/J teacher sources")
    input_pins = {name: _file_spec(path) for name, path in {
        "protocol": protocol, "preserved": preserved_path, "prior_endpoint": prior_path,
        "common_start": common_path, "manifest": manifest_path,
    }.items()}
    existing_path = output / "endpoint-lock.json"
    if existing_path.exists():
        with runner_context(models):
            existing = pipeline._load_lock(existing_path, kind="endpoint")
            if (list(existing["endpoints"]) != list(models) or existing["training_root"] != str(run)
                    or existing["repo_root"] != str(repo)
                    or any(existing["input_pins"].get(key) != value for key, value in input_pins.items())
                    or existing["targets"] != {key: _file_spec(value) for key, value in targets.items()}):
                raise ValueError("existing endpoint lock belongs to different screen inputs")
            _verify_screen_files(existing)
            return existing
    preserved = pipeline._read_json(preserved_path)
    prior_relative = prior_path.relative_to(repo).as_posix()
    if preserved.get("files", {}).get(prior_relative) != input_pins["prior_endpoint"]["sha256"]:
        raise ValueError("prior endpoint metadata differs from the preserved pre-screen snapshot")
    # Pin old evaluation metadata without parsing any prior outcome report.
    for relative, digest in preserved["files"].items():
        if relative.startswith("reports/contrast-factorial-v1/pipeline-v1/"):
            path = pipeline._verify_file(repo, relative, digest, "preserved evaluation metadata")
            input_pins[relative] = _file_spec(path)
    prior = pipeline._load_lock(prior_path, kind="endpoint")
    if prior.get("repo_root") != str(repo):
        raise ValueError("prior endpoint lock belongs to another repository")
    old_approval_path = _check_spec(prior["approval"], "prior approval")
    approval, _, base, _, _ = pipeline._validate_approval(repo, old_approval_path)
    manifest = pipeline._read_json(manifest_path)
    pipeline._verify_manifest(repo, manifest_path, manifest, approval, old_approval_path)
    if prior["manifest"] != _file_spec(manifest_path):
        raise ValueError("training manifest differs from the prior endpoint lock")
    common = pipeline._read_json(common_path)
    ordinary_keys = {"checkpoint_id", "trainable_weights_sha256", "cpu_rng_sha256", "cuda_rng_sha256"}
    pipeline._validate_start_evidence({key: common.get(key) for key in ordinary_keys}, approval)
    if (set(common) != ordinary_keys | {"optimizer_state_entries", "optimizer_initial_sha256"}
            or common.get("optimizer_state_entries") != 0
            or not isinstance(common.get("optimizer_initial_sha256"), str)
            or len(common["optimizer_initial_sha256"]) != 64):
        raise ValueError("common start lacks fresh matched Adam evidence")
    suite_path = pipeline._repo_path(repo, manifest["arms"]["B"]["suite_file"], "B training suite")
    target_specs = {}
    for teacher, path in targets.items():
        value = load_verified_targets(path, suite_path)
        wanted = {"Q": "Qwen/Qwen3.5-4B", "J": "jev-1.13.0"}[teacher]
        if value["teacher"]["model"] != wanted:
            raise ValueError("teacher target source exchanged or mismatched")
        target_specs[teacher] = _file_spec(path)
    endpoints, sources = {}, {}
    if "BASE" in models:
        endpoints["BASE"] = {"checkpoint": str(base), "checkpoint_id": approval["base_checkpoint_id"],
                             "tree_sha256": approval["base_checkpoint_sha256"], "exposure": None}
    for arm in models:
        if arm == "BASE":
            continue
        endpoint, arm_sources, pins = _validate_arm(repo, arm, run / arm / "step-0400", manifest,
                                                  pipeline.sha256_file(manifest_path), approval, common, target_specs)
        endpoints[arm] = endpoint
        sources.update(arm_sources)
        input_pins.update(pins)
    selected = [arm for arm in models if arm != "BASE"]
    first = endpoints[selected[0]]
    for arm in selected[1:]:
        current = endpoints[arm]
        for key in ("family_ids", "replay_ids"):
            if current[key] != first[key]:
                raise ValueError("arm family/replay exposures are not position-matched")
        for key in ("schedule_sha256", "replay_position_sha256", "prepared_prompt_fingerprint", "runtime"):
            if current["fingerprint"].get(key) != first["fingerprint"].get(key):
                raise ValueError(f"arm {key} differs from matched screen")
    for arm in selected:
        endpoints[arm].pop("family_ids")
        endpoints[arm].pop("replay_ids")
    partitions = copy.deepcopy(prior["partitions"])
    if set(partitions) != {"assessment", "calibration", "legacy_calibration", "legacy_retention", "legacy_manifest"}:
        raise ValueError("prior endpoint partition inventory differs")
    for name, spec in partitions.items():
        path = pipeline._repo_path(repo, spec["path"], name)
        if any(part.lower() in {"test", "tests", "finaltest"} for part in path.relative_to(repo).parts):
            raise ValueError("TEST/Jev paths remain quarantined")
        pipeline._verify_file(repo, spec["path"], spec["sha256"], name)
    # Pin the actually imported evaluation sources, also for fixtures using a tiny repository.
    code_root = Path(__file__).resolve().parents[1]
    for relative in EVALUATION_SOURCES:
        sources[str(code_root / relative)] = pipeline.sha256_file(code_root / relative)
    new_approval = {**approval, "study": STUDY, "arms": selected,
                    "contract_path": protocol.relative_to(repo).as_posix(), "contract_sha256": pipeline.sha256_file(protocol),
                    "evaluation_scope": SCOPE, "authorization": "User-authorized bounded screen recorded in PROTOCOL.md",
                    "prior_endpoint": input_pins["prior_endpoint"], "preserved": input_pins["preserved"]}
    approval_path = output / "study-approval.json"
    if approval_path.exists() and pipeline._read_json(approval_path) != new_approval:
        raise ValueError("pre-existing study approval differs")
    pipeline._atomic_json(approval_path, new_approval)
    reporting = {**pipeline.REPORTING_PROTOCOL, "selection": "Only explicitly selected complete update-400 arms",
                 "evaluation_scope": SCOPE, "comparisons": [f"{a}-{b}" for a, b in COMPARISONS]}
    value = {"version": 1, "study": STUDY, "kind": "endpoint", "status": "locked",
             "created_utc": pipeline._utc(), "lock_path": str(existing_path), "repo_root": str(repo),
             "training_root": str(run), "gpu_lock": str(output / "gpu.lock"),
             "shared_study_lock": str(screen / "STUDY.lock"), "approval": _file_spec(approval_path),
             "manifest": _file_spec(manifest_path), "input_pins": input_pins, "source_pins": sources,
             "endpoints": endpoints, "selected_arms": selected, "excluded_arms": [arm for arm in ARMS if arm not in selected],
             "common_start": common, "targets": target_specs, "training_suite": str(suite_path),
             "partitions": partitions, "evaluation_scope": SCOPE,
             "reporting": {"source": str(Path(__file__).resolve()), "sha256": pipeline.sha256_file(__file__), "protocol": reporting}}
    value["lock_sha256"] = pipeline.canonical_hash(value)
    with runner_context(models):
        _verify_screen_files(value)
    pipeline._atomic_json(existing_path, value)
    return value


def run_evaluation(endpoint_lock_path, output_root, *, engine_loader=None, json_loader=None):
    """Hold the same lock as training/Qwen and invoke original versioned stages."""
    raw = pipeline._read_json(endpoint_lock_path)
    models = _models(list(raw.get("endpoints", {})))
    expected_lock = Path(raw["repo_root"]) / "reports" / STUDY / "STUDY.lock"
    if raw.get("shared_study_lock") != str(expected_lock):
        raise ValueError("endpoint names a different shared study lock")
    with training.study_lock(expected_lock), runner_context(models):
        endpoint = pipeline._load_lock(endpoint_lock_path, kind="endpoint")
        _verify_screen_files(endpoint)
        result = pipeline._run_evaluation_locked(endpoint_lock_path, output_root,
                                                 engine_loader=engine_loader, json_loader=json_loader)
        _verify_screen_files(endpoint)
        return result


def estimate_pairwise_effects(arm_values, categories, *, bootstrap_samples=2000, seed=42):
    """Category-macro differences, resampling whole families jointly across arms."""
    if (not arm_values or not set(arm_values) <= set(ARMS) or not categories
            or type(bootstrap_samples) is not int or bootstrap_samples < 1 or type(seed) is not int):
        raise ValueError("invalid paired family population or bootstrap settings")
    ids = set(categories)
    for values in arm_values.values():
        if set(values) != ids or any(type(x) not in (int, float) or not math.isfinite(x) for x in values.values()):
            raise ValueError("arms must contain the same finite family population")
    strata = defaultdict(list)
    for identity in sorted(ids):
        category = categories[identity]
        if not isinstance(category, str) or not category:
            raise ValueError("family categories must be nonempty strings")
        strata[category].append(identity)
    groups = [strata[key] for key in sorted(strata)]
    def means(sample):
        return {arm: statistics.mean(statistics.mean(values[key] for key in group) for group in sample)
                for arm, values in arm_values.items()}
    comparisons = [(a, b) for a, b in COMPARISONS if a in arm_values and b in arm_values]
    point = means(groups)
    draws = {f"{a}-{b}": [] for a, b in comparisons}
    rng = random.Random(seed)
    for _ in range(bootstrap_samples):
        sample = means([[rng.choice(group) for _ in group] for group in groups])
        for a, b in comparisons:
            draws[f"{a}-{b}"].append(sample[a] - sample[b])
    return {"means": point, "effects": {f"{a}-{b}": {"estimate": point[a] - point[b],
            "ci95": [_percentile(draws[f"{a}-{b}"], 0.025), _percentile(draws[f"{a}-{b}"], 0.975)]}
            for a, b in comparisons}, "families": len(ids), "categories": len(strata),
            "family_counts": {key: len(group) for key, group in strata.items()},
            "bootstrap_samples": bootstrap_samples, "seed": seed,
            "excluded_comparisons": [f"{a}-{b}" for a, b in COMPARISONS if (a, b) not in comparisons],
            "aggregation": "Equal family weight within category; equal category weight; same sampled families in every arm.",
            "limitation": "Descriptive corpus uncertainty for one training seed on inspected development data."}


def build_final_report(output_root):
    """Revalidate saved evidence, then report original metrics and screen contrasts."""
    output = Path(output_root).resolve()
    raw = pipeline._read_json(output / "endpoint-lock.json")
    models = _models(list(raw.get("endpoints", {})))
    with runner_context(models):
        endpoint = pipeline._load_lock(output / "endpoint-lock.json", kind="endpoint")
        _verify_screen_files(endpoint)
        calibration = pipeline._load_lock(output / "calibration-lock.json", kind="calibration")
        fits = pipeline._validate_calibration_lock(endpoint, calibration)
        records = pipeline._validate_evaluation_status(endpoint, calibration, pipeline._read_json(output / "evaluation-status.json"))
        assessment = pipeline._load_partition(endpoint, "assessment", pipeline._read_json)
        retention = pipeline._load_partition(endpoint, "legacy_retention", pipeline._read_json)
        results = {}
        for model in models:
            temperatures = fits[model]["temperatures"]
            variants = {}
            for variant in VARIANTS:
                parts = {}
                for name, suite in (("assessment", assessment), ("retention", retention)):
                    saved = records[model][name]
                    rows = ([record["row"] for record in saved] if variant == "raw" else
                            pipeline._calibrated_rows(saved, temperatures, variant))
                    parts[name] = summarize_factorial(suite, rows)
                    parts[name + "_relation_rows"] = pipeline._relation_rows(suite, rows)
                variants[variant] = parts
            results[model] = {"checkpoint_id": endpoint["endpoints"][model]["checkpoint_id"],
                              "temperatures": temperatures, "variants": variants,
                              "compute": endpoint["endpoints"][model].get("exposure")}
        effects = {}
        for variant in VARIANTS:
            values = {arm: results[arm]["variants"][variant]["assessment"]["primary"] for arm in endpoint["selected_arms"]}
            first = next(iter(values.values()))
            if any(value["family_categories"] != first["family_categories"] for value in values.values()):
                raise ValueError("paired family category populations differ")
            effects[variant] = estimate_pairwise_effects({arm: value["family_values"] for arm, value in values.items()},
                                                       first["family_categories"])
        report = {"version": 1, "study": STUDY, "status": "complete", "generated_utc": pipeline._utc(),
                  "evaluation_scope": SCOPE, "selected_arms": endpoint["selected_arms"], "excluded_arms": endpoint["excluded_arms"],
                  "baseline_included": "BASE" in models, "endpoint_lock_sha256": endpoint["lock_sha256"],
                  "calibration_lock_sha256": calibration["lock_sha256"], "protocol": endpoint["reporting"]["protocol"],
                  "models": results, "pairwise_effects": effects,
                  "limitations": ["Previously inspected assessment, including teacher-audit families, is development data.",
                                  "One training seed; family intervals describe corpus uncertainty, not training-seed variance.",
                                  "Original TEST/Jev quarantine remains unopened; independent confirmation requires a later decision.",
                                  "Synthetic programs limit external validity; these results do not confirm a winner."],
                  "decision": "No automatic promotion or follow-on experiment is authorized by this report."}
        _verify_screen_files(endpoint)
        final = output / "final-report"
        pipeline._atomic_json(final / "report.json", report)
        lines = ["# Calibrated screen v1 — EXPLORATORY development results", "", SCOPE, "",
                 f"Completed selected arms: {', '.join(endpoint['selected_arms'])}.",
                 f"Excluded arms: {', '.join(endpoint['excluded_arms']) or 'none'}.",
                 "All selected endpoints and all temperature fits were locked before development inference.", "",
                 "| Model | RAW pair | Global T pair | Per-primitive T pair |", "|---|---:|---:|---:|"]
        for model in models:
            values = [results[model]["variants"][v]["assessment"]["primary"]["category_macro"] for v in VARIANTS]
            lines.append(f"| {model} | " + " | ".join(f"{value:.6f}" for value in values) + " |")
        lines += ["", "| Calibration | Comparison | Difference | 95% paired family interval |", "|---|---|---:|---:|"]
        for variant, value in effects.items():
            for name, effect in value["effects"].items():
                lines.append(f"| {variant} | {name} | {effect['estimate']:.6f} | [{effect['ci95'][0]:.6f}, {effect['ci95'][1]:.6f}] |")
        lines += ["", "Full RAW/global/per-primitive category, primitive, Score K, pair, NLL, Brier, Unknown, Score MAE, "
                  "confidence counts/coverage/risk and retention metrics are in report.json; raw logits and responses are retained per stage.", "",
                  *report["limitations"], "", report["decision"], ""]
        (final / "RESULTS.md").write_text("\n".join(lines))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("lock", "evaluate", "report"))
    parser.add_argument("--training-root", type=Path, default=REPO_ROOT / "checkpoints/calibrated-screen-v1/run-v1")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "reports/calibrated-screen-v1/evaluation")
    parser.add_argument("--arms", nargs="+", choices=("BASE", *ARMS))
    parser.add_argument("--targets", action="append", default=[], metavar="SOURCE=PATH")
    parser.add_argument("--manifest", type=Path, default=training.MANIFEST)
    parser.add_argument("--protocol", type=Path, default=training.PROTOCOL)
    parser.add_argument("--preserved-artifacts", type=Path, default=REPO_ROOT / "reports/calibrated-screen-v1/preserved-artifacts.json")
    parser.add_argument("--prior-endpoint-lock", type=Path, default=REPO_ROOT / "reports/contrast-factorial-v1/pipeline-v1/endpoint-lock.json")
    parser.add_argument("--common-start", type=Path, default=REPO_ROOT / "reports/calibrated-screen-v1/common-start.json")
    args = parser.parse_args(argv)
    if args.command == "lock":
        if not args.arms:
            parser.error("lock requires explicit --arms")
        targets = {}
        for value in args.targets:
            key, separator, path = value.partition("=")
            if key not in {"Q", "J"} or not separator or not path or key in targets:
                parser.error("--targets requires unique Q=PATH or J=PATH")
            targets[key] = Path(path)
        with training.study_lock(REPO_ROOT / "reports" / STUDY / "STUDY.lock"):
            result = create_endpoint_lock(repo_root=REPO_ROOT, manifest_path=args.manifest,
                training_root=args.training_root, output_root=args.output, arms=args.arms,
                protocol_path=args.protocol, preserved_path=args.preserved_artifacts,
                prior_endpoint_lock=args.prior_endpoint_lock, common_start_path=args.common_start, targets=targets)
    elif args.command == "evaluate":
        result = run_evaluation(args.output / "endpoint-lock.json", args.output)
    else:
        result = build_final_report(args.output)
    print(json.dumps({key: result[key] for key in ("study", "status")}, indent=2))
    return result


if __name__ == "__main__":
    main()
