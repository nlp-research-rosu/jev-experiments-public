"""Pinned source conversion and context-safe preparation for judgment training.

Raw releases are content-addressed in data/raw-v0.2/source-lock.json. Preparation
is offline once those files exist; optional --download fetches only locked URLs.
The source case is split/deduplicated before related views are constructed.
"""
from __future__ import annotations

import argparse
import bisect
import copy
import csv
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "judgment-corpus-v0.2"
SEED = "openjev-context-split-v0.2-20260919"
BOOLQ_REV = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
CLINC_REV = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
PAWS_REV = "161ece9501cf0a11f3e48bd356eaa82de46d6a09"
TASKSOURCE_REV = "1dee7ed51b87880e37882c2cbed2d50bd114e0df"
BANKING_REV = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SPLITS = ("train", "validation", "calibration", "test")
TRAIN_QUOTAS = {"boolq": 4000, "paws": 4000, "clinc150": 6500, "tasksource/banking77": 1500,
                "wine_quality": 3000, "oracle_policy": 1000}
WINE_FEATURES = ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def text_key(value):
    return "text:" + digest(normalize_text(value))


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _case(dataset, revision, split, row, task, state, question, target, contexts, **provenance):
    return {
        "id": f"{dataset}/{split}/{row}", "original_split": split,
        "context_keys": sorted(set(contexts)),
        "example": {"state": state, "question": question, "target": target},
        "provenance": {"dataset": dataset, "revision": revision, "row": row, "task": task,
                       "original_split": split, **provenance},
    }


def convert_boolq(row, split, index):
    if type(row.get("answer")) is not bool:
        raise ValueError("BoolQ answer must be boolean")
    passage = _nonempty(row.get("passage"), "passage")
    question = _nonempty(row.get("question"), "question")
    instructions = "Using the passage, is the answer to this question yes? Question: " + question
    return _case("boolq", BOOLQ_REV, split, index, "passage_grounded_yes_no",
                 {"passage": passage}, {"type": "noul", "instructions": instructions,
                 "criteria": {"true": "The answer to the stated question, given the passage, is yes.",
                              "false": "The answer to the stated question, given the passage, is no."}},
                 {"truth": row["answer"]}, [text_key(passage)], label_origin="human-outcome")


def convert_paws(row, split, index):
    if type(row.get("label")) is not int or row["label"] not in (0, 1):
        raise ValueError("PAWS label must be 0 or 1")
    a = _nonempty(row.get("sentence1"), "sentence1")
    b = _nonempty(row.get("sentence2"), "sentence2")
    # Token-multiset groups also catch the source word-permutation families.
    # They do not assert equivalence or create a training relation.
    contexts = [text_key(a), text_key(b)]
    for text in (a, b):
        words = re.findall(r"\w+", normalize_text(text))
        contexts.append("wordbag:" + digest(" ".join(sorted(words))))
    return _case("paws", PAWS_REV, split, index, "sentence_meaning_equivalence",
                 {"sentence1": a, "sentence2": b},
                 {"type": "noul", "instructions": "Do sentence1 and sentence2 express the same meaning?",
                  "criteria": {"true": "The two sentences are paraphrases with the same meaning.",
                               "false": "The two sentences differ in meaning."}},
                 {"truth": row["label"] == 1}, contexts, source_id=row["id"], label_origin="human-outcome")


def convert_clinc(row, split, index, ontology, *, k=4):
    if len(row) != 2 or row[1] not in ontology or row[1] == "oos":
        raise ValueError("CLINC row requires an in-scope ontology label")
    if k not in (4, 8) or len(set(ontology)) < k:
        raise ValueError("Choice candidate count must be four or eight")
    text, label = _nonempty(row[0], "utterance"), row[1]
    key = digest(normalize_text(text))
    negatives = sorted(set(ontology) - {label}, key=lambda name: digest(SEED + key + name))[:k - 1]
    candidates = sorted([label, *negatives], key=lambda name: digest("option-order" + key + name))
    return _case("clinc150", CLINC_REV, split, index, "intent_among_declared_candidates",
                 {"utterance": text}, {"type": "choice",
                 "instructions": "Which intent among the supplied candidates best matches the utterance? "
                                 "The correct intent is included in this restricted candidate set.",
                 "criteria": {name: "The utterance expresses the intent: " + name.replace("_", " ") + "."
                              for name in candidates}}, {"choice": label}, [text_key(text)],
                 label_origin="human-outcome", candidate_selection={
                     "method": "gold-plus-hash-ranked-ontology-negatives-v1", "seed": SEED,
                     "count": k, "source_ontology_size": len(ontology), "full_ontology": "clinc150",
                     "not_original_full_classification": True})


def convert_tasksource_banking(row, split, index, originals):
    if row.get("task") != "banking77":
        raise ValueError("unapproved Tasksource task")
    header, separator, text = row["inputs"].partition("\n")
    candidates = re.findall(r'"([^"\n]+)"', header)
    target = row["targets"].removesuffix(".")
    if not separator or len(candidates) not in (4, 8) or len(set(candidates)) != len(candidates):
        raise ValueError("Tasksource requires one explicit four/eight-option classification header")
    if not header.startswith("With no explanation, label the following with either ") or target not in candidates:
        raise ValueError("unsupported Tasksource classification format")
    original = originals.get(normalize_text(text))
    if not original or original["label"] != target:
        raise ValueError("Tasksource target does not match original BANKING77 gold")
    # Official original test wins even if a derivative release calls a row train.
    effective_split = "test" if original["split"] == "test" else split
    case = _case("tasksource/banking77", TASKSOURCE_REV, effective_split, index, "banking_intent_among_declared_candidates",
                 {"utterance": text}, {"type": "choice",
                 "instructions": "Which banking intent among the supplied candidates best matches the utterance? "
                                 "The correct intent is included in this restricted candidate set.",
                 "criteria": {name: "The utterance expresses the banking intent: " + name.replace("_", " ") + "."
                              for name in candidates}}, {"choice": target}, [text_key(text)],
                 label_origin="human-outcome", tasksource_task="banking77", tasksource_split=split,
                 component_revision=BANKING_REV, component_dataset="PolyAI-LDN/task-specific-datasets/banking_data",
                 component_row=original["row"], component_original_split=original["split"],
                 candidate_selection={"method": "preserved-tasksource-declared-candidates",
                                      "count": len(candidates), "source_ontology_size": 77,
                                      "full_ontology": "banking77", "not_original_full_classification": True})
    case["id"] = f"tasksource/banking77/{split}/{index}"
    return case


def convert_wine(row, color, index):
    quality = int(row["quality"])
    if not 0 <= quality <= 10 or color not in ("red", "white"):
        raise ValueError("invalid Wine Quality rating/type")
    state = {"wine_type": color, "measurements": {name: float(row[name]) for name in WINE_FEATURES}}
    band = 0 if quality <= 5 else (1 if quality == 6 else 2)
    return _case("wine_quality", "uci-186-2009-content-addressed", "unsplit", f"{color}/{index}",
                 "expert_sensory_quality_three_bands", state,
                 {"type": "score", "instructions": "Predict the expert sensory quality band of this Portuguese "
                  "Vinho Verde wine from its physicochemical measurements, ordered from lower to higher quality.",
                  "criteria": ["Expert sensory quality rating from 0 through 5 on the original 0–10 scale.",
                               "Expert sensory quality rating of 6 on the original 0–10 scale.",
                               "Expert sensory quality rating from 7 through 10 on the original 0–10 scale."]},
                 {"level_index": band}, ["structured:" + digest(canonical(state))],
                 original_quality=quality, label_origin="expert-ordinal-rating-binned",
                 target_mapping="0..5 -> 0; 6 -> 1; 7..10 -> 2")


def synthetic_policy_case(family, observed, thresholds, index):
    labels = {"stock": ("items in stock", "inventory level"),
              "queue": ("waiting requests", "queue load"),
              "completion": ("completed work units", "completion level"),
              "latency": ("response time in milliseconds", "latency severity")}
    if family not in labels or len(thresholds) != 3 or thresholds != sorted(set(thresholds)):
        raise ValueError("invalid synthetic policy family or ordered thresholds")
    field, meaning = labels[family]
    a, b, c = thresholds
    rubric = [f"{meaning.capitalize()}: {field} is less than {a}.",
              f"{meaning.capitalize()}: {field} is at least {a} and less than {b}.",
              f"{meaning.capitalize()}: {field} is at least {b} and less than {c}.",
              f"{meaning.capitalize()}: {field} is at least {c}."]
    state = {field: observed}
    return _case("oracle_policy", "oracle-thresholds-v1", "test" if family == "latency" else "unsplit",
                 f"{family}/{index}", f"synthetic_{family}_rubric", state,
                 {"type": "score", "instructions": f"Apply the supplied four-level {meaning} rubric to the measured value.",
                  "criteria": rubric}, {"level_index": bisect.bisect_right(thresholds, observed)},
                 ["oracle:" + digest(canonical([family, observed, thresholds]))],
                 label_origin="deterministic-oracle", synthetic=True, family=family,
                 oracle_rule="bisect_right(strictly increasing thresholds, measured value)", thresholds=thresholds,
                 heldout_task_family=family == "latency")


def make_bundle(case, split, group_id):
    original = copy.deepcopy(case["example"])
    examples, relations = [original], []
    if original["question"]["type"] == "noul":
        negative = copy.deepcopy(original)
        whole_question = original["question"]["instructions"]
        wording = "Is the answer NO to the following complete yes/no judgment? " if split == "test" else \
                  "Is it false that the answer is YES to the following complete yes/no judgment? "
        negative["question"]["instructions"] = wording + "[" + whole_question + "]"
        criteria = original["question"]["criteria"]
        negative["question"]["criteria"] = {"true": criteria["false"], "false": criteria["true"]}
        negative["target"]["truth"] = not original["target"]["truth"]
        examples.append(negative)
        relations.append({"kind": "complement", "left": 0, "right": 1,
                          "transformation_id": "scoped-no-v1" if split == "test" else "scoped-false-yes-v1",
                          "verification_status": "verified-by-construction",
                          "justification": "Complement of the entire original judgment with exchanged binary criteria."})
    if case["provenance"]["dataset"] == "paws":
        swapped = copy.deepcopy(original)
        state = original["state"]
        swapped["state"] = {"sentence1": state["sentence2"], "sentence2": state["sentence1"]}
        examples.append(swapped)
        relations.append({"kind": "invariant", "left": 0, "right": 2,
                          "transformation_id": "paraphrase-sentence-swap-v1",
                          "verification_status": "verified-by-symmetry",
                          "justification": "Sentence meaning equivalence is symmetric for both positive and negative pairs."})
    return {"id": case["id"], "group_id": group_id, "examples": examples, "relations": relations,
            "provenance": {**copy.deepcopy(case["provenance"]), "context_hashes": case["context_keys"],
                           "assigned_split": split}}


def _semantic_key(case):
    example = case["example"]
    if case["provenance"]["dataset"] == "paws":
        semantic = ["sentence_meaning_equivalence", sorted(normalize_text(v) for v in example["state"].values())]
    elif case["provenance"]["dataset"] == "tasksource/banking77":
        # Different sampled candidate sets are repeated views of one original label.
        semantic = ["banking77", normalize_text(example["state"]["utterance"])]
    elif case["provenance"]["dataset"] == "clinc150":
        semantic = ["clinc150", normalize_text(example["state"]["utterance"])]
    else:
        semantic = [example["state"], example["question"]]
    return digest(canonical(semantic))


def _split_for_group(group_id, original_splits):
    number = int(digest(SEED + group_id)[:16], 16) / 16**16
    if "test" in original_splits:
        return "test"
    if any(split in {"validation", "val", "dev"} for split in original_splits):
        return "validation" if number < 0.5 else "calibration"
    return "train" if number < 0.8 else ("validation" if number < 0.87 else ("calibration" if number < 0.94 else "test"))


def partition_cases(cases):
    """Deduplicate and split union-connected contexts, prioritizing official holdouts."""
    cases = sorted(cases, key=lambda case: case["id"])
    parent = list(range(len(cases)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    context_owner, semantic_owner = {}, {}
    duplicate_sets = defaultdict(list)
    for index, case in enumerate(cases):
        semantic = _semantic_key(case)
        duplicate_sets[semantic].append(index)
        if semantic in semantic_owner:
            union(index, semantic_owner[semantic])
        semantic_owner[semantic] = index
        for context in case["context_keys"]:
            if context in context_owner:
                union(index, context_owner[context])
            context_owner[context] = index
    groups = defaultdict(list)
    for index in range(len(cases)):
        groups[find(index)].append(index)
    stats = {"converted_cases": len(cases), "duplicate_rows_removed": 0, "conflicting_rows_removed": 0,
             "cross_source_context_groups": 0, "groups_promoted_to_official_holdout": 0,
             "context_groups_before_sampling": len(groups)}
    remove = set()
    for indices in duplicate_sets.values():
        targets = {canonical(cases[i]["example"]["target"]) for i in indices}
        if len(targets) > 1:
            remove.update(indices)
            stats["conflicting_rows_removed"] += len(indices)
        else:
            # Retain the original heldout record when source duplicates have different splits.
            ordered = sorted(indices, key=lambda i: (
                {"test": 0, "validation": 1, "val": 1, "dev": 1}.get(cases[i]["original_split"], 2), cases[i]["id"]))
            remove.update(ordered[1:])
            stats["duplicate_rows_removed"] += len(ordered) - 1
    splits = {name: [] for name in SPLITS}
    for indices in groups.values():
        contexts = sorted({context for i in indices for context in cases[i]["context_keys"]})
        group_id = digest(canonical(contexts))
        originals = {cases[i]["original_split"] for i in indices}
        assigned = _split_for_group(group_id, originals)
        sources = {cases[i]["provenance"]["dataset"] for i in indices}
        stats["cross_source_context_groups"] += len(sources) > 1
        stats["groups_promoted_to_official_holdout"] += (
            "train" in originals and bool(originals & {"test", "validation", "val", "dev"}))
        for index in indices:
            if index not in remove:
                splits[assigned].append(make_bundle(cases[index], assigned, group_id))
    for rows in splits.values():
        rows.sort(key=lambda row: row["id"])
    stats["eligible_by_split"] = {split: len(rows) for split, rows in splits.items()}
    return splits, stats


def _balanced_take(rows, limit):
    strata = defaultdict(list)
    for row in rows:
        target = canonical(row["examples"][0]["target"])
        strata[(row["provenance"]["task"], target)].append(row)
    for values in strata.values():
        values.sort(key=lambda row: digest(SEED + "sample" + row["id"]))
    result = []
    depth = 0
    while len(result) < limit:
        added = False
        for key in sorted(strata):
            if depth < len(strata[key]) and len(result) < limit:
                result.append(strata[key][depth])
                added = True
        if not added:
            break
        depth += 1
    return result


def sample_splits(splits):
    selected = {}
    for split, rows in splits.items():
        by_source = defaultdict(list)
        for row in rows:
            by_source[row["provenance"]["dataset"]].append(row)
        chosen = []
        for source, candidates in sorted(by_source.items()):
            limit = TRAIN_QUOTAS[source] if split == "train" else (250 if split == "test" else 150)
            chosen.extend(_balanced_take(candidates, limit))
        # Hash order interleaves domains reproducibly without label-sorted training.
        selected[split] = sorted(chosen, key=lambda row: digest(SEED + "order" + row["id"]))
    return selected


def write_corpus(output, splits, sources, stats, *, extra=None):
    output = Path(output)
    group_split, context_split, ids = {}, {}, set()
    manifest = {"version": VERSION, "seed": SEED, "sources": sources, "preparation": stats,
                "splits": {}, "target_contract": {"choice": "choice:string", "score": "level_index:int", "noul": "truth:bool"},
                "no_confidence_targets": True, **(extra or {})}
    for split in SPLITS:
        rows = splits.get(split, [])
        primitive, tasks, datasets, relations, labels, option_counts = (Counter() for _ in range(6))
        groups = set()
        for row in rows:
            if row["id"] in ids:
                raise ValueError("duplicate ID or cross-split leakage")
            ids.add(row["id"])
            group = row["group_id"]
            if group in group_split and group_split[group] != split:
                raise ValueError("group leakage across splits")
            group_split[group] = split
            groups.add(group)
            for context in row["provenance"]["context_hashes"]:
                if context in context_split and context_split[context] != split:
                    raise ValueError("context leakage across splits")
                context_split[context] = split
            datasets[row["provenance"]["dataset"]] += 1
            tasks[row["provenance"]["task"]] += 1
            first = row["examples"][0]
            primitive[first["question"]["type"]] += 1
            labels[row["provenance"]["dataset"] + "/" + canonical(first["target"])] += 1
            for example in row["examples"]:
                kind = example["question"]["type"]
                if kind != "noul":
                    option_counts[f"{kind}/{len(example['question']['criteria'])}"] += 1
            relations.update(relation["kind"] for relation in row["relations"])
        content = "".join(canonical(row) + "\n" for row in rows)
        manifest["splits"][split] = {"bundles": len(rows), "examples": sum(len(row["examples"]) for row in rows),
            "groups": len(groups), "primitive_bundles": dict(primitive), "datasets": dict(datasets), "tasks": dict(tasks),
            "labels": dict(labels), "candidate_counts": dict(option_counts), "relations": dict(relations),
            "sha256": digest(content), "bytes": len(content.encode("utf-8"))}
    # Validate the whole corpus before touching its output files.
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output / f"{split}.jsonl").write_text("".join(canonical(row) + "\n" for row in splits.get(split, [])), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest


def _parquet_rows(path):
    import pyarrow.parquet as pq
    file = pq.ParquetFile(path)
    for batch in file.iter_batches(batch_size=16384):
        yield from batch.to_pylist()


def load_locked_sources(raw, *, download=False):
    raw = Path(raw)
    lock = json.loads((raw / "source-lock.json").read_text())
    for entry in lock["files"]:
        file = raw / entry["name"]
        if not file.exists() and download:
            temporary = file.with_suffix(file.suffix + ".partial")
            urllib.request.urlretrieve(entry["url"], temporary)
            temporary.replace(file)
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"source checksum mismatch: {file.name}")
    return lock


def load_cases(raw, lock):
    raw = Path(raw)
    cases, rejected, scanned = [], Counter(), Counter()
    hashes = {entry["name"]: entry["sha256"] for entry in lock["files"]}

    def append(case, file):
        case["provenance"]["source_file"] = file.name
        case["provenance"]["source_sha256"] = hashes[file.name]
        cases.append(case)

    for split in ("train", "validation"):
        file = raw / f"google--boolq--data--{split}-00000-of-00001.parquet"
        for index, row in enumerate(_parquet_rows(file)):
            scanned["boolq"] += 1
            append(convert_boolq(row, split, index), file)
    file = raw / "clinc-full.json"
    clinc = json.loads(file.read_text())
    ontology = sorted({row[1] for row in clinc["train"]})
    for split, rows in clinc.items():
        for index, row in enumerate(rows):
            scanned["clinc150"] += 1
            if split.startswith("oos_"):
                rejected["clinc150/oos_not_in_bounded_candidate_task"] += 1
                continue
            append(convert_clinc(row, split, index, ontology, k=4 if index % 2 == 0 else 8), file)
    for split in ("train", "validation", "test"):
        file = raw / f"paws--labeled_final--{split}-00000-of-00001.parquet"
        for index, row in enumerate(_parquet_rows(file)):
            scanned["paws"] += 1
            append(convert_paws(row, split, index), file)
    originals = {}
    for split in ("train", "test"):
        file = raw / f"PolyAI-LDN--task-specific-datasets--banking_data--{split}.csv"
        for index, row in enumerate(csv.DictReader(io.StringIO(file.read_text()))):
            key = normalize_text(row["text"])
            old = originals.get(key)
            if old and old["label"] != row["category"]:
                raise ValueError("conflicting original BANKING77 labels")
            originals[key] = {"label": row["category"], "split": split, "row": index}
    for file in sorted(raw.glob("tasksource--*--data--*.parquet")):
        split = file.name.split("--data--")[1].split("-")[0]
        for index, row in enumerate(_parquet_rows(file)):
            scanned["tasksource_all_rows_scanned"] += 1
            if row["task"] != "banking77":
                continue
            scanned["tasksource/banking77"] += 1
            try:
                case = convert_tasksource_banking(row, split, f"{file.name}/{index}", originals)
            except ValueError as error:
                rejected[f"tasksource/banking77/{error}"] += 1
                continue
            if not set(case["example"]["question"]["criteria"]).issubset({x["label"] for x in originals.values()}):
                rejected["tasksource/banking77/candidate_outside_original_ontology"] += 1
                continue
            append(case, file)
    file = raw / "wine-quality.zip"
    with zipfile.ZipFile(file) as archive:
        for color in ("red", "white"):
            content = archive.read(f"winequality-{color}.csv").decode("utf-8")
            for index, row in enumerate(csv.DictReader(io.StringIO(content), delimiter=";")):
                scanned["wine_quality"] += 1
                append(convert_wine(row, color, index), file)
    synthetic_rows = []
    for family in ("stock", "queue", "completion", "latency"):
        for index in range(1000):
            # Independent hash-derived integers, with deliberate boundary examples.
            number = int(digest(SEED + family + str(index)), 16)
            a = 5 + number % 91
            b = a + 5 + (number // 101) % 91
            c = b + 5 + (number // 10009) % 91
            boundary_values = [a - 1, a, b - 1, b, c - 1, c]
            observed = boundary_values[index % 6] if index % 3 == 0 else (number // 1000003) % (c + 101)
            case = synthetic_policy_case(family, observed, [a, b, c], index)
            cases.append(case)
            synthetic_rows.append({"family": family, "index": index, "observed": observed, "thresholds": [a, b, c]})
    scanned["oracle_policy"] = len(synthetic_rows)
    return cases, {"scanned": dict(scanned), "rejected": dict(rejected),
                   "synthetic_spec_sha256": digest(canonical(synthetic_rows))}, {"clinc150": ontology,
                   "banking77": sorted({x["label"] for x in originals.values()})}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw-v0.2"))
    parser.add_argument("--output", type=Path, default=Path("data/processed-v0.2"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    lock = load_locked_sources(args.raw, download=args.download)
    cases, read_stats, ontologies = load_cases(args.raw, lock)
    splits, stats = partition_cases(cases)
    sampled = sample_splits(splits)
    stats.update(read_stats)
    stats["train_source_quotas"] = TRAIN_QUOTAS
    stats["eligible_by_source_split"] = {split: dict(Counter(row["provenance"]["dataset"] for row in rows))
                                         for split, rows in splits.items()}
    stats["sampling_removed_by_split"] = {split: len(splits[split]) - len(sampled[split]) for split in SPLITS}
    manifest = write_corpus(args.output, sampled, lock["sources"], stats, extra={
        "source_lock_sha256": hashlib.sha256((args.raw / "source-lock.json").read_bytes()).hexdigest(),
        "preparation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_files": lock["files"], "source_ontologies": ontologies,
        "selection": "Source quotas, then round-robin task/gold strata; SHA256 ranks within strata.",
        "split_policy": "Context-connected components before views; official test wins, official dev reserved "
                        "for validation/calibration; other groups hash to 80/7/7/6 percent train/validation/calibration/test.",
        "limitations": lock["limitations"]})
    print(json.dumps({split: info["bundles"] for split, info in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
