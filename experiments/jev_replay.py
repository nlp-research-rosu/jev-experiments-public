"""Prepare or replay frozen Noul, Choice and Score diagnostics against TypeSafe."""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.jev_archive import DEFAULT_DIRECTORY, DEFAULT_MODEL, VALIDATION_POLICY_VERSION, APIError, ArchiveClient
from experiments.revised_metrics import relation_metrics, summarize, target_label, validate_suite

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
KEY_FILE = Path(".secrets/typesafe_api_key")
NLL_FLOOR = 1e-12
SMOKE_IDS = (
    "action/support/claim_only",
    "action/support/live_success",
    "action/support/timeout_call",
    "payment.explicit_unconfirmed",
    "payment.confirmed_without_claim",
    "payment.unverified_record",
    "payment.conflicting_records",
    "payment.unchanged_repeat",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_key(path=KEY_FILE):
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and path.exists():
        key = path.read_text()
    key = (key or "").strip()
    if not key or any(c.isspace() for c in key) or not key.isascii():
        raise ValueError("Set TYPESAFE_API_KEY or place the single API key in the private key file.")
    return key


def make_payload(case, model):
    return {"model": model, **copy.deepcopy(case["request"])}


def post(payload, key, *, archive=DEFAULT_DIRECTORY, fresh=False):
    """Compatibility entry point; every live response now goes into the archive."""
    return ArchiveClient(key, archive, fresh=fresh)(payload)


def probability_view(probabilities, expected):
    p = copy.deepcopy(probabilities)
    if (
        not isinstance(p, dict) or expected not in p
        or any(type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values())
        or not math.isclose(sum(p.values()), 1, rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError("Invalid probability distribution")
    peak = max(p.values())
    winners = [k for k, v in p.items() if abs(v - peak) <= 1e-12]
    winner = winners[0] if len(winners) == 1 else None
    return {
        "expected": expected, "predicted": winner, "probabilities": p,
        "correct": winner == expected, "ambiguous_top": len(winners) != 1,
        "nll": -math.log(max(p[expected], NLL_FLOOR)),
        "zero_target_probability": p[expected] == 0,
        "brier": sum((v - float(k == expected)) ** 2 for k, v in p.items()),
        "max_probability": peak,
    }


def _remote_probabilities(probabilities, labels):
    """Validate complete vectors and retain the declared cent-rounding policy."""
    if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
        raise ValueError("Response probability classes differ from request")
    if not probabilities or any(
        type(v) not in (float, int) or not math.isfinite(v) or not 0 <= v <= 1
        for v in probabilities.values()
    ):
        raise ValueError("Invalid response probability")
    annotations = {}
    total = math.fsum(probabilities.values())
    if not math.isclose(total, 1, rel_tol=0, abs_tol=1e-6):
        if total <= 0 or abs(total - 1) > 0.020000001 or not _cent_grid(probabilities.values()):
            raise ValueError("Probability sum exceeds the cent-rounding policy")
        annotations.update(
            raw_probabilities=copy.deepcopy(probabilities), reported_probability_sum=total,
            probability_normalized=True,
        )
        probabilities = {key: value / total for key, value in probabilities.items()}
    return probabilities, annotations


def _cent_grid(values):
    return all(abs(value - round(value * 100) / 100) <= 1e-9 for value in values)


def answer_view(question, expected, answer):
    if not isinstance(answer, dict) or answer.get("type") != question["type"]:
        raise ValueError("Response primitive does not match request")
    annotations = {"primitive": question["type"]}
    if question["type"] == "noul":
        p = answer.get("noul")
        if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid Noul probability")
        probabilities = {"false": 1 - p, "true": p}
    elif question["type"] == "choice":
        raw_probabilities = answer.get("probabilities")
        probabilities, rounding = _remote_probabilities(raw_probabilities, question["criteria"])
        annotations.update(rounding)
        choice, confidence = answer.get("choice"), answer.get("confidence")
        if (
            choice not in raw_probabilities or raw_probabilities[choice] < max(raw_probabilities.values()) - 1e-6
            or type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1
        ):
            raise ValueError("Invalid Choice winner or confidence")
        annotations["confidence"] = confidence
    elif question["type"] == "score":
        criteria = question.get("criteria")
        if not isinstance(criteria, list) or not criteria or any(
            not isinstance(value, (str, dict, list)) for value in criteria
        ):
            raise ValueError("Score requires an ordered nonempty rubric")
        if type(expected) is not int or not 0 <= expected < len(criteria):
            raise ValueError("Score target must be a declared integer index")
        legend = {str(index): value for index, value in enumerate(criteria)}
        if answer.get("legend") != legend:
            raise ValueError("Response Score legend differs from request")
        probabilities, rounding = _remote_probabilities(answer.get("probabilities"), legend)
        annotations.update(rounding)
        score, confidence = answer.get("score"), answer.get("confidence")
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= len(criteria) - 1:
            raise ValueError("Invalid returned Score mean")
        if type(confidence) not in (float, int) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid Score confidence")
        mean = math.fsum(int(key) * value for key, value in probabilities.items())
        # Let e_i be unrounded minus reported probability and m the normalized
        # reported mean. Since the latent probabilities sum to 1, the latent
        # mean minus m is sum((i-m)*e_i). Cent rounding bounds |e_i| by .005.
        # Add .005 for independently rounded returned score and 1e-6 numerics.
        tolerance = 0.005001
        if _cent_grid(answer["probabilities"].values()):
            tolerance += 0.005 * math.fsum(abs(index - mean) for index in range(len(criteria)))
        if abs(score - mean) > tolerance:
            raise ValueError("Returned Score mean disagrees with the full probability vector")
        error = abs(mean - expected)
        annotations.update(
            score_mean=mean, score_target=expected, score_absolute_error=error,
            score_normalized_absolute_error=error / max(len(criteria) - 1, 1),
            returned_score=score, confidence=confidence, score_mean_tolerance=tolerance,
        )
    else:
        raise ValueError("This diagnostic supports Noul, Choice and Score")
    result = probability_view(probabilities, target_label(expected))
    return {**result, **annotations}


def compare_local(remote_rows, local_rows):
    local = {(r["case_id"], r["question_id"]): r["prediction"] for r in local_rows}
    if len(local) != len(local_rows) or not remote_rows:
        raise ValueError("Need unique local references and nonempty remote predictions")
    disagreements, variations, local_views, remote_views = [], [], [], []
    for row in remote_rows:
        key = row["case_id"], row["question_id"]
        old, new = local.get(key), row["prediction"]
        if old is None or old["expected"] != new["expected"] or old["probabilities"].keys() != new["probabilities"].keys():
            raise ValueError("Comparison population, labels or answer spaces differ")
        a = probability_view(old["probabilities"], old["expected"])
        b = probability_view(new["probabilities"], new["expected"])
        local_views.append(a)
        remote_views.append(b)
        tv = 0.5 * sum(abs(a["probabilities"][k] - b["probabilities"][k]) for k in a["probabilities"])
        variations.append(tv)
        if a["predicted"] != b["predicted"]:
            disagreements.append({"case_id": key[0], "question_id": key[1], "local": a, "jev": b, "tv": tv})
    return {
        "questions": len(remote_rows), "label_disagreements": len(disagreements),
        "mean_total_variation": statistics.mean(variations),
        "local_correct": sum(r["correct"] for r in local_views),
        "jev_correct": sum(r["correct"] for r in remote_views),
        "local_nll_capped": statistics.mean(r["nll"] for r in local_views),
        "jev_nll_capped": statistics.mean(r["nll"] for r in remote_views),
        "disagreements": disagreements,
    }


def run_suite(suite, output, *, model, call, metadata=None, local_references=None):
    validate_suite(suite)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "prepared-not-sent", "started_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT, "requested_model": model, "returned_models": [],
        "requests_planned": len(suite["cases"]), "requests_completed": 0,
        "questions_planned": sum(len(c["expected"]) for c in suite["cases"]),
        "usage": {"input_tokens": 0, "output_tokens": 0}, "metadata": metadata or {},
        "new_usage": {"input_tokens": 0, "output_tokens": 0},
        "cache_hits": 0, "live_attempts": 0, "archive_records": [],
        "normalized_probability_vectors": 0,
        "metric_policy": {
            "validation_policy_version": VALIDATION_POLICY_VERSION,
            "choice_probability_normalization": "Validate keys and finite values in [0,1]. If sum differs from 1 by more than 1e-6, normalize only a positive total within 0.020000001 of 1 with every value on a 0.01 grid within 1e-9. Preserve and flag raw vectors; local probability_view stays strict.",
            "score_probability_normalization": "Same complete-vector and cent-grid policy as Choice. Exact string indices 0 through k-1 and the original indexed rubric legend are required.",
            "score_mean": "Compute mean and MAE from the complete normalized probability vector; preserve returned_score independently. Mean agreement tolerance is 0.005001 plus 0.005*sum(abs(i-mean)) only for cent-grid vectors, accounting for rounded scalar and bins; this is a formatting bound, not calibration.",
            "nll": "From returned probabilities only; floor at 1e-12, including local comparisons. No raw Jev logits.",
            "brier": "Sum over all categories; binary is twice the scalar convention.",
            "latency": "Client call time, including cache reads when reused; per-response archive metadata identifies cache hits. Not local GPU kernel time.",
            "usage": "usage totals describe observations; new_usage counts validated fresh responses only, not verified billing or failed-response usage.",
            "confidence": "Stored as returned; not interpreted as probability of correctness.",
            "gold": "Authored diagnostic contract; Jev answers are observations, not replacement labels.",
        },
    }
    with (output / "requests.jsonl").open("x") as stream:
        for case in suite["cases"]:
            stream.write(json.dumps({"case_id": case["id"], "request": make_payload(case, model)}, allow_nan=False) + "\n")
    write_json(output / "report.json", report)
    if call is None:
        return report
    rows, lookup, times = [], {}, []
    report["status"] = "running"
    try:
        for case in suite["cases"]:
            payload = make_payload(case, model)
            started = time.perf_counter()
            try:
                response = call(payload)
            finally:
                archive_record = copy.deepcopy(getattr(call, "last_record", None))
                if archive_record is not None:
                    report["archive_records"].append({"case_id": case["id"], **archive_record})
                    report["cache_hits" if archive_record["cache_hit"] else "live_attempts"] += 1
                else:
                    report["live_attempts"] += 1
            elapsed = (time.perf_counter() - started) * 1000
            # Preserve successful HTTP responses even when the semantic schema check fails.
            with (output / "responses.jsonl").open("a") as stream:
                stream.write(json.dumps({
                    "case_id": case["id"], "response": response, "elapsed_ms": elapsed,
                    "received_utc": datetime.now(timezone.utc).isoformat(),
                    "archive": archive_record,
                }, allow_nan=False) + "\n")
            returned_model = response.get("model")
            if not isinstance(returned_model, str) or not returned_model:
                raise ValueError("Missing returned model version")
            if returned_model not in report["returned_models"]:
                report["returned_models"].append(returned_model)
            if len(report["returned_models"]) != 1:
                raise ValueError("Model version changed during replay")
            if not isinstance(response.get("answers"), dict) or set(response["answers"]) != set(payload["questions"]):
                raise ValueError("Missing or unexpected answer IDs")
            case_rows = []
            for qid, question in payload["questions"].items():
                prediction = answer_view(question, case["expected"][qid], response["answers"][qid])
                case_rows.append({
                    **{key: case[key] for key in ("family_id", "domain", "variant")},
                    "case_id": case["id"], "question_id": qid, "prediction": prediction,
                    "layout": case.get("layout", "unspecified"),
                })
            usage = response.get("usage", {})
            if any(type(usage.get(k)) is not int or usage[k] < 0 for k in report["usage"]):
                raise ValueError("Invalid token usage")
            rows.extend(case_rows)
            report["normalized_probability_vectors"] += sum(bool(r["prediction"].get("probability_normalized")) for r in case_rows)
            lookup.update({(r["case_id"], r["question_id"]): r["prediction"] for r in case_rows})
            times.append(elapsed)
            for key in report["usage"]:
                report["usage"][key] += usage[key]
                if archive_record is None or not archive_record["cache_hit"]:
                    report["new_usage"][key] += usage[key]
            report["requests_completed"] += 1
            write_json(output / "report.json", report)
        relations = [{**r, "metrics": relation_metrics(r, lookup)} for r in suite["relations"]]
        write_json(output / "relations.json", relations)
        report.update(
            status="complete", summary=summarize(rows, relations),
            zero_target_probabilities=sum(r["prediction"]["zero_target_probability"] for r in rows),
            latency_ms={"median": statistics.median(times), "min": min(times), "max": max(times)},
            comparisons={name: compare_local(rows, values) for name, values in (local_references or {}).items()},
        )
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
        if isinstance(error, APIError):
            report["http_status"] = error.status
        if not isinstance(error, Exception):
            raise
    finally:
        write_json(output / "predictions.json", rows)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path("data/semantic-contrasts-v1"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--archive", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--fresh", action="store_true", help="Bypass reuse and preserve a separate live sample")
    parser.add_argument("--key-file", type=Path, default=KEY_FILE)
    parser.add_argument("--run", action="store_true", help="Send live requests; without this flag prepare only")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Use the full suite instead of the eight-case smoke set")
    selection.add_argument("--case-id", action="append")
    args = parser.parse_args()
    suite = read_json(args.suite / "suite.json")
    manifest = read_json(args.suite / "manifest.json")
    digest = sha(args.suite / "suite.json")
    if digest != manifest["suite_sha256"]:
        parser.error("Frozen suite hash mismatch")
    ids = set(args.case_id or SMOKE_IDS) if not args.all else {c["id"] for c in suite["cases"]}
    if not ids <= {c["id"] for c in suite["cases"]}:
        parser.error("Unknown case ID; custom suites need --all or explicit --case-id")
    suite["cases"] = [c for c in suite["cases"] if c["id"] in ids]
    suite["relations"] = [r for r in suite["relations"] if all(r[e]["case_id"] in ids for e in ("left", "right"))]
    key = None
    if args.run:
        try:
            key = load_key(args.key_file)
        except (ValueError, OSError):
            parser.error("API key unavailable; configure TYPESAFE_API_KEY or the private key file. Do not paste it in chat.")
    metadata = {"suite_sha256": digest, "source_sha256": sha(__file__), "local_reference_sha256": {}}
    references = {}
    old_report_path = Path("reports/semantic-contrasts-v1/report.json")
    if old_report_path.exists() and read_json(old_report_path)["manifest"]["suite_sha256"] == digest:
        for name in ("untrained", "trained"):
            path = old_report_path.parent / f"{name}-predictions.json"
            references[name] = read_json(path)
            metadata["local_reference_sha256"][str(path)] = sha(path)
    result = run_suite(
        suite, args.output, model=args.model,
        call=ArchiveClient(key, args.archive, fresh=args.fresh) if args.run else None,
        metadata=metadata, local_references=references,
    )
    print(json.dumps({key: result[key] for key in ("status", "requests_planned", "requests_completed", "questions_planned")}))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
