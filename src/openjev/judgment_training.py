"""Complete-question supervision and verified paired consistency for judgments."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class TrainingGroup:
    primitive: str
    indices: tuple[int, ...]
    criteria: object
    target: dict


@dataclass(frozen=True)
class PreparedBundle:
    id: str
    source: str
    prompts: list[list[int]]
    kinds: list[int]
    groups: list[TrainingGroup]
    relations: list[dict]


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("target probabilities must be finite numbers in [0,1]")
    return float(value)


def target_vector(group):
    target, kind = group.target, group.primitive
    if not isinstance(target, dict):
        raise ValueError("target must be an object")
    if kind == "noul":
        if set(target) == {"truth"} and type(target["truth"]) is bool:
            p = float(target["truth"])
        elif set(target) == {"probability_true"}:
            p = _number(target["probability_true"])
        else:
            raise ValueError("Noul requires boolean truth or probability_true")
        if len(group.indices) != 1:
            raise ValueError("Noul requires one binary logit")
        return [1 - p, p]
    if kind not in ("choice", "score"):
        raise ValueError("unknown training primitive")
    k = len(group.criteria)
    if k != len(group.indices) or k == 0:
        raise ValueError("incomplete candidate group")
    if kind == "choice" and set(target) == {"choice"}:
        if not isinstance(target["choice"], str) or target["choice"] not in group.criteria:
            raise ValueError("unknown choice target")
        vector = [float(c == target["choice"]) for c in group.criteria]
    elif kind == "score" and set(target) == {"level_index"}:
        index = target["level_index"]
        if type(index) is not int or not 0 <= index < k:
            raise ValueError("invalid level index")
        vector = [float(i == index) for i in range(k)]
    elif set(target) == {"probabilities"}:
        raw = target["probabilities"]
        if kind == "choice":
            if not isinstance(raw, dict) or set(raw) != set(group.criteria):
                raise ValueError("target probability keys must match choices")
            vector = [_number(raw[name]) for name in group.criteria]
        else:
            if not isinstance(raw, list) or len(raw) != k:
                raise ValueError("target vector must match levels")
            vector = [_number(p) for p in raw]
    else:
        raise ValueError("target does not match primitive")
    if not math.isclose(sum(vector), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("target distribution must sum to one")
    return vector


def bundle_loss(logits, groups, relations, *, consistency_weight=0.1):
    if not groups or logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("finite one-dimensional logits and nonempty groups required")
    if not math.isfinite(consistency_weight) or consistency_weight < 0:
        raise ValueError("consistency weight must be nonnegative")
    if sorted(i for g in groups for i in g.indices) != list(range(len(logits))):
        raise ValueError("groups must cover each logit exactly once")
    supervised, probabilities, targets = [], [], []
    for group in groups:
        q = target_vector(group)
        targets.append(q)
        scores = logits[list(group.indices)]
        if group.primitive == "noul":
            supervised.append(F.binary_cross_entropy_with_logits(scores[0], scores.new_tensor(q[1])))
            p = scores[0].sigmoid()
            probabilities.append(torch.stack((1 - p, p)))
        else:
            target = scores.new_tensor(q)
            supervised.append(-(target * scores.log_softmax(-1)).sum())
            probabilities.append(scores.softmax(-1))
    terms = []
    for relation in relations:
        left, right = relation.get("left"), relation.get("right")
        if (
            type(left) is not int
            or type(right) is not int
            or not (0 <= left < len(groups) and 0 <= right < len(groups))
            or left == right
        ):
            raise ValueError("invalid relation example indices")
        a, b = groups[left], groups[right]
        p, q = probabilities[left], probabilities[right]
        ta, tb = targets[left], targets[right]
        if relation.get("kind") == "complement":
            if a.primitive != "noul" or b.primitive != "noul" or abs(ta[1] + tb[1] - 1) > 1e-6:
                raise ValueError("complement relation needs inverse binary targets")
            q = q.flip(0)
        elif relation.get("kind") == "invariant":
            if a.primitive != b.primitive or len(p) != len(q):
                raise ValueError("invariant views have different answer spaces")
            if a.primitive == "choice":
                if set(a.criteria) != set(b.criteria):
                    raise ValueError("invariant choices must name the same outcomes")
                order = [list(b.criteria).index(name) for name in a.criteria]
                q = q[order]
                tb = [tb[i] for i in order]
            if any(abs(x - y) > 1e-6 for x, y in zip(ta, tb, strict=True)):
                raise ValueError("invariant relation has conflicting labels")
        else:
            raise ValueError("unsupported consistency relation")
        terms.append(0.5 * (p - q).square().sum())
    sup = torch.stack(supervised).mean()
    consistency = torch.stack(terms).mean() if terms else logits.sum() * 0
    total = sup + consistency_weight * consistency
    if not torch.isfinite(total):
        raise ValueError("non-finite training loss")
    return {"total": total, "supervised": sup, "consistency": consistency, "probabilities": probabilities}


def prepare_bundle(engine, bundle):
    prompts, kinds, groups = [], [], []
    if not bundle.get("examples"):
        raise ValueError("bundle needs examples")
    for example in bundle["examples"]:
        compiled, tokens, readouts = engine.prepare(
            {"state": example["state"], "questions": {"answer": example["question"]}}
        )
        if len(compiled.questions) != 1:
            raise ValueError("training example must contain one complete question")
        question = compiled.questions[0]
        group = TrainingGroup(
            question.primitive,
            tuple(i + len(prompts) for i in question.unit_indices),
            question.criteria,
            example["target"],
        )
        target_vector(group)
        groups.append(group)
        prompts.extend(tokens)
        kinds.extend(readouts)
    source = bundle.get("provenance", {}).get("dataset", "unknown")
    return PreparedBundle(bundle["id"], str(source), prompts, kinds, groups, bundle.get("relations", []))


def load_bundles(path):
    records = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"invalid JSON at {path}:{number}") from exc
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("need nonempty unique bundle IDs")
    return records


@torch.inference_mode()
def evaluate_bundles(engine, prepared):
    engine.model.eval()
    total, nll, brier, consistency = 0, [], [], []
    correct = 0
    per_source = {}
    canonical = {"questions": 0, "correct": 0, "nll": 0.0, "brier": 0.0, "by_source": {}}
    residuals = {"complement": [], "invariant": []}
    raw = []
    for bundle in prepared:
        logits = engine.model.score_prompts(bundle.prompts, bundle.kinds, unit_batch_size=engine.unit_batch_size)
        result = bundle_loss(logits, bundle.groups, bundle.relations)
        for index, (group, p) in enumerate(zip(bundle.groups, result["probabilities"], strict=True)):
            y = target_vector(group)
            probs = p.cpu().tolist()
            predicted = max(range(len(probs)), key=probs.__getitem__)
            expected = max(range(len(y)), key=y.__getitem__)
            ok = predicted == expected
            total += 1
            correct += ok
            z = logits[list(group.indices)].float()
            loss = (
                F.binary_cross_entropy_with_logits(z[0], z.new_tensor(y[1]))
                if group.primitive == "noul"
                else -(z.new_tensor(y) * z.log_softmax(-1)).sum()
            )
            nll.append(loss.item())
            brier.append(sum((a - b) ** 2 for a, b in zip(y, probs, strict=True)))
            stats = per_source.setdefault(bundle.source, {"questions": 0, "correct": 0})
            stats["questions"] += 1
            stats["correct"] += ok
            if index == 0:
                canonical["questions"] += 1
                canonical["correct"] += ok
                canonical["nll"] += nll[-1]
                canonical["brier"] += brier[-1]
                stats = canonical["by_source"].setdefault(bundle.source, {"questions": 0, "correct": 0, "nll": 0.0})
                stats["questions"] += 1
                stats["correct"] += ok
                stats["nll"] += nll[-1]
        for relation in bundle.relations:
            i, j = relation["left"], relation["right"]
            p, q = result["probabilities"][i], result["probabilities"][j]
            if relation["kind"] == "complement":
                q = q.flip(0)
            elif bundle.groups[i].primitive == "choice":
                q = q[[list(bundle.groups[j].criteria).index(k) for k in bundle.groups[i].criteria]]
            residuals[relation["kind"]].append((p - q).abs().max().item())
        consistency.append(result["consistency"].item())
        raw.append(
            {
                "id": bundle.id,
                "logits": logits.cpu().tolist(),
                "probabilities": [p.cpu().tolist() for p in result["probabilities"]],
            }
        )
        if len(raw) % 200 == 0:
            print(f"Evaluated {len(raw)}/{len(prepared)} bundles", flush=True)
    if not total:
        raise ValueError("empty evaluation")
    canonical["accuracy"] = canonical["correct"] / canonical["questions"]
    canonical["nll"] /= canonical["questions"]
    canonical["brier"] /= canonical["questions"]
    for stats in canonical["by_source"].values():
        stats["nll"] /= stats["questions"]
        stats["accuracy"] = stats["correct"] / stats["questions"]
    return {
        "metric_population": "all question views including augmentations; canonical reports first original view only",
        "bundles": len(prepared),
        "questions": total,
        "accuracy": correct / total,
        "nll": sum(nll) / total,
        "brier": sum(brier) / total,
        "mean_consistency_loss": sum(consistency) / len(consistency),
        "by_source": per_source,
        "canonical": canonical,
        "consistency_residuals": {
            kind: {
                "pairs": len(values),
                "mean": sum(values) / len(values),
                "p95": sorted(values)[math.ceil(0.95 * len(values)) - 1],
                "max": max(values),
            }
            for kind, values in residuals.items()
            if values
        },
        "predictions": raw,
    }
