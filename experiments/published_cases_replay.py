"""Replay TypeSafe's selected display bundles; not its unreleased full workflow eval."""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from openjev.judgment_cli import load_judgment_engine, read_json
from openjev.judgments import compile_request


def parse_snapshot(text):
    prefix = "__VIEWER_DATA__("
    if not text.startswith(prefix):
        raise ValueError("unexpected public snapshot wrapper")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate snapshot key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"nonfinite snapshot constant {value}")

    value, end = json.JSONDecoder(object_pairs_hook=unique, parse_constant=invalid).raw_decode(text[len(prefix) :])
    if text[len(prefix) + end :].strip() not in (");", ")") or not isinstance(value, dict):
        raise ValueError("snapshot has extra executable content or wrong root")
    return value


def indexed(items, index):
    if type(index) is not int or not 0 <= index < len(items):
        raise ValueError("invalid published index")
    return items[index]


def extract_bundles(snapshot):
    evaluation = snapshot["eval"]
    examples = {item["case_id"]: item for item in evaluation["examples"]}
    records, seen = [], set()
    for case_id, case in evaluation["cases"].items():
        trace = case["models"]["typesafe"]
        for node in trace["nodes"]:
            if node.get("ran") is not True or not node.get("questions"):
                continue
            key = (evaluation["id"], case_id, node["node"])
            if key in seen:
                raise ValueError("duplicate published node")
            seen.add(key)
            request = {
                "state": copy.deepcopy(indexed(evaluation["documents"], node["doc"])),
                "questions": {
                    qid: copy.deepcopy(indexed(evaluation["questions"], index))
                    for qid, index in node["questions"].items()
                },
            }
            compile_request(request)
            if set(node["answers"]) != set(request["questions"]):
                raise ValueError("published answer keys do not match node questions")
            records.append(
                {
                    "id": json.dumps(key),
                    "workflow": evaluation["id"],
                    "case_id": case_id,
                    "node": node["node"],
                    "example": examples[case_id],
                    "request": request,
                    "historical_model": trace["model"],
                    "historical_answers": copy.deepcopy(node["answers"]),
                    "reference_answers": copy.deepcopy(case.get("reference_answers", {}).get(node["node"], {})),
                    "reference_descriptions": copy.deepcopy(case.get("references", [])),
                    "stored_workflow_decisions": copy.deepcopy(trace.get("decisions", {})),
                    "source_doc_index": node["doc"],
                    "source_question_indices": copy.deepcopy(node["questions"]),
                }
            )
    return records


def labels(question):
    if question["type"] == "noul":
        return ["false", "true"]
    if question["type"] == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return list(question["criteria"])


def normalized(raw, names):
    if not isinstance(raw, dict) or set(raw) != set(names):
        raise ValueError("probability options differ from the replayed question")
    if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in raw.values()):
        raise ValueError("invalid probability")
    total = sum(raw.values())
    if total <= 0 or abs(total - 1) > 0.05:
        raise ValueError("probability total exceeds display-rounding tolerance")
    return {name: raw[name] / total for name in names}, total


def unique_winner(probabilities):
    maximum = max(probabilities.values())
    winners = [name for name, value in probabilities.items() if abs(value - maximum) <= 1e-12]
    return winners[0] if len(winners) == 1 else None


def answer_view(question, answer):
    kind = question["type"]
    if answer.get("type") != kind:
        raise ValueError("answer primitive does not match question")
    raw = {"false": 1 - answer["noul"], "true": answer["noul"]} if kind == "noul" else answer["probabilities"]
    p, total = normalized(raw, labels(question))
    label = unique_winner(p)
    if kind == "choice":
        label = answer["choice"]
        if label not in p:
            raise ValueError("published choice outside question options")
    result = {"label": label, "probabilities": p, "original_probability_sum": total, "max_probability": max(p.values())}
    if kind == "score":
        result["reported_score"] = answer["score"]
        result["expected_score"] = sum(int(name) * value for name, value in p.items())
    return result


def reference_view(question, entry):
    unavailable = {"label": None, "probabilities": None, "probabilistic": False}
    if not entry or not entry.get("sets"):
        return {**unavailable, "status": "missing"}
    if entry.get("type") != question["type"]:
        return {**unavailable, "status": "incompatible", "reason": "primitive differs"}
    names = labels(question)
    vectors, sources, totals = [], [], []
    try:
        for item in entry["sets"]:
            if item.get("probabilities") is not None:
                p, total = normalized(item["probabilities"], names)
                sources.append("probabilities")
                totals.append(total)
            else:
                value = item.get("value")
                if question["type"] == "noul":
                    if type(value) is not bool:
                        raise ValueError("missing binary reference label")
                    value = "true" if value else "false"
                elif question["type"] == "score" and type(value) is int:
                    value = str(value)
                if value not in names:
                    raise ValueError("reference label outside question options")
                p = {name: float(name == value) for name in names}
                sources.append("label")
                totals.append(None)
            vectors.append(p)
    except (ValueError, TypeError) as exc:
        return {**unavailable, "status": "incompatible", "reason": str(exc)}
    pooled = {name: statistics.mean(p[name] for p in vectors) for name in names}
    label = unique_winner(pooled)
    probabilistic = all(s == "probabilities" for s in sources)
    result = {
        "status": "ok" if label is not None else "tied",
        "label": label,
        "probabilities": pooled,
        "probabilistic": probabilistic,
        "sets": len(vectors),
        "original_probability_sums": totals,
        "member_labels": [item.get("value") for item in entry["sets"]],
        "rule": "mean_probabilities"
        if probabilistic
        else "label_votes"
        if all(s == "label" for s in sources)
        else "mixed",
    }
    if question["type"] == "score":
        result["expected_score"] = sum(int(name) * value for name, value in pooled.items())
    return result


def tv(a, b):
    if a.keys() != b.keys():
        raise ValueError("cannot compare differing answer spaces")
    return 0.5 * sum(abs(a[k] - b[k]) for k in a)


def summarize(rows, variants):
    def group(items):
        eligible = [r for r in items if r["reference"]["label"] is not None]
        soft = [r for r in items if r["reference"]["probabilistic"]]
        result = {
            "questions": len(items),
            "reference_status": dict(Counter(r["reference"]["status"] for r in items)),
            "determinate_reference_questions": len(eligible),
            "probabilistic_reference_questions": len(soft),
            "models": {},
        }
        for model in ["jev", *variants]:
            completed = [r for r in items if model in r["outputs"]]
            if len(completed) != len(items):
                continue
            matches = sum(r["outputs"][model]["label"] == r["reference"]["label"] for r in eligible)
            result["models"][model] = {
                "reference_label_matches": matches,
                "reference_label_agreement": matches / len(eligible) if eligible else None,
                "mean_tv_to_probabilistic_reference": statistics.mean(
                    tv(r["outputs"][model]["probabilities"], r["reference"]["probabilities"]) for r in soft
                )
                if soft
                else None,
                "jev_label_matches": sum(r["outputs"][model]["label"] == r["outputs"]["jev"]["label"] for r in items),
                "mean_tv_to_jev": statistics.mean(
                    tv(r["outputs"][model]["probabilities"], r["outputs"]["jev"]["probabilities"]) for r in items
                ),
            }
        return result

    result = {"overall": group(rows), "by_workflow": {}, "by_primitive": {}, "by_case": {}}
    for field, destination in (("workflow", "by_workflow"), ("primitive", "by_primitive"), ("case_key", "by_case")):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result[destination] = {key: group(items) for key, items in groups.items()}
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_records(root):
    lock = read_json(root / "source-lock.json")
    records = []
    for source in lock["sources"]:
        filename = source["workflow"] + "-cases.js"
        if sha(root / filename) != source["files"][filename]["sha256"]:
            raise ValueError("snapshot hash mismatch")
        records.extend(extract_bundles(parse_snapshot((root / filename).read_text())))
    return records


def run(args, report):
    records = load_records(args.snapshot)
    write(args.output / "requests.json", records)
    report["request_sha256"] = sha(args.output / "requests.json")
    rows, lookup = [], {}
    for record in records:
        for qid, question in record["request"]["questions"].items():
            row = {
                "id": json.dumps([record["id"], qid]),
                "bundle_id": record["id"],
                "workflow": record["workflow"],
                "case_key": json.dumps([record["workflow"], record["case_id"]]),
                "case_id": record["case_id"],
                "case_name": record["example"]["name"],
                "selection_label": record["example"]["label"],
                "node": record["node"],
                "qid": qid,
                "primitive": question["type"],
                "question": question,
                "reference": reference_view(question, record["reference_answers"].get(qid)),
                "reference_raw": record["reference_answers"].get(qid),
                "outputs": {"jev": answer_view(question, record["historical_answers"][qid])},
            }
            rows.append(row)
            lookup[record["id"], qid] = row
    report.update(
        bundles=len(records),
        cases=len({row["case_key"] for row in rows}),
        questions=len(rows),
        summaries=summarize(rows, []),
        historical_models=sorted({r["historical_model"] for r in records}),
    )
    write(args.output / "comparisons.json", rows)
    write(args.output / "report.json", report)
    if args.prepare_only:
        report["status"] = "prepared"
        write(args.output / "report.json", report)
        return
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    completed = []
    report["runs"] = {}
    for variant in ("untrained", "trained"):
        started = time.perf_counter()
        engine = load_judgment_engine(
            checkpoint=args.checkpoint if variant == "trained" else None,
            backend="fla",
            unit_batch_size=args.batch_size,
            max_input_tokens=args.max_tokens,
        )
        report["runs"][variant] = {
            "model_id": engine.model_id,
            "load_seconds": time.perf_counter() - started,
            "nodes": [],
        }
        engine.evaluate(records[0]["request"], details=True)
        torch.cuda.reset_peak_memory_stats()
        for index, record in enumerate(records):
            response = engine.evaluate(record["request"], details=True)
            if response["usage"]["max_unit_tokens"] > args.max_tokens:
                raise RuntimeError("token bound not enforced")
            for qid, answer in response["answers"].items():
                lookup[record["id"], qid]["outputs"][variant] = answer_view(record["request"]["questions"][qid], answer)
            with (args.output / (variant + "-responses.jsonl")).open("a") as stream:
                stream.write(
                    json.dumps({"bundle_id": record["id"], "response": response}, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
            entry = {"bundle_id": record["id"], "usage": response["usage"], "elapsed_ms": response["elapsed_ms"]}
            report["runs"][variant]["nodes"].append(entry)
            report["progress"] = {"variant": variant, "completed_bundles": index + 1, "total_bundles": len(records)}
            write(args.output / "report.json", report)
            write(args.output / "comparisons.json", rows)
            print(
                f"{variant} {index + 1}/{len(records)} {record['workflow']} {record['case_id']} {record['node']}: {response['usage']['max_unit_tokens']} tokens, {response['elapsed_ms']:.0f}ms",
                flush=True,
            )
        completed.append(variant)
        report["runs"][variant]["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["summaries"] = summarize(rows, completed)
        write(args.output / "report.json", report)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    report.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("reference/evals/typesafe-2026-09-20"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/judgment-full-v0.2/final"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=12288)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1 or args.max_tokens < 1:
        parser.error("choose a new output directory and positive resource bounds")
    args.output.mkdir(parents=True)
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot": str(args.snapshot),
        "snapshot_lock_sha256": sha(args.snapshot / "source-lock.json"),
        "source_sha256": {str(p): sha(p) for p in [Path(__file__), *Path("src/openjev").glob("judgment*.py")]},
        "scope": "Selected published TypeSafe node bundles replayed on their recorded states; not dynamic workflow execution or the complete official evaluation. No training or prompt tuning.",
        "reference_policy": "Mean supplied probabilities, or hard-label votes where probabilities absent; explicit ties, missing refs and incompatible schemas excluded from label agreement. Votes are not calibrated probabilities.",
        "execution": {
            "backend": "fla",
            "dtype": "BF16 backbone, FP32 heads/LoRA",
            "cached": True,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_tokens,
            "truncation": False,
            "merged": False,
        },
        "packages": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
    }
    try:
        run(args, report)
    except BaseException as exc:
        report.update(status="failed", error=repr(exc))
        write(args.output / "report.json", report)
        raise


if __name__ == "__main__":
    main()
