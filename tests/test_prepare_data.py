"""Source conversion and leakage tests using source-shaped, hand-checked records."""
import copy
import json

import pytest

from openjev import prepare_data as prep

BOOLQ = {
    "question": "do iran and afghanistan speak the same language",
    "answer": True,
    "passage": "Persian is an official language in Iran and Afghanistan.",
}
PAWS = {
    "id": 1,
    "sentence1": "Although interchangeable, the body pieces on the 2 cars are not similar.",
    "sentence2": "Although similar, the body parts are not interchangeable on the 2 cars.",
    "label": 0,
}


def test_boolq_preserves_passage_and_boolean_and_exact_complement():
    case = prep.convert_boolq(BOOLQ, "train", 5)
    bundle = prep.make_bundle(case, "test", "group")
    left, right = bundle["examples"]
    assert left["state"] == {"passage": BOOLQ["passage"]}
    assert left["target"] == {"truth": True}
    assert right["target"] == {"truth": False}
    assert left["question"]["instructions"] in right["question"]["instructions"]
    assert right["question"]["criteria"] == {
        "true": left["question"]["criteria"]["false"],
        "false": left["question"]["criteria"]["true"],
    }
    assert bundle["relations"][0]["kind"] == "complement"
    assert (bundle["relations"][0]["left"], bundle["relations"][0]["right"]) == (0, 1)
    assert "task" not in left["state"] and "row" not in left["state"]


@pytest.mark.parametrize("answer", ["False", 0, None])
def test_boolq_does_not_coerce_malformed_boolean(answer):
    with pytest.raises(ValueError):
        prep.convert_boolq({**BOOLQ, "answer": answer}, "train", 0)


def test_paws_negative_pair_swaps_sentences_without_changing_equivalence_label():
    bundle = prep.make_bundle(prep.convert_paws(PAWS, "train", 0), "train", "g")
    first, negative, swapped = bundle["examples"]
    assert first["target"] == swapped["target"] == {"truth": False}
    assert negative["target"] == {"truth": True}
    assert swapped["state"] == {"sentence1": PAWS["sentence2"], "sentence2": PAWS["sentence1"]}
    assert any(r["kind"] == "invariant" and r["left"] == 0 and r["right"] == 2 for r in bundle["relations"])


def test_clinc_declares_subset_and_keeps_target_with_four_or_eight_options():
    ontology = ["card_arrival", "cancel_transfer", "card_not_working", "card_acceptance", "refund", "balance", "top_up", "cash"]
    for k in [4, 8]:
        case = prep.convert_clinc(["I think my card is broken", "card_not_working"], "train", 0, ontology, k=k)
        question = case["example"]["question"]
        assert len(question["criteria"]) == k
        assert case["example"]["target"] == {"choice": "card_not_working"}
        assert "card_not_working" in question["criteria"]
        assert "among the supplied" in question["instructions"]
        assert case["provenance"]["candidate_selection"]["source_ontology_size"] == 8
    with pytest.raises(ValueError):
        prep.convert_clinc(["outside", "oos"], "oos_train", 0, ontology)


def test_tasksource_banking_checks_original_gold_and_original_test_membership():
    row = {"inputs": 'With no explanation, label the following with either "card_arrival", "card_not_working", "cancel_transfer" or "card_acceptance".\nI think my card is broken', "targets": "card_not_working.", "task": "banking77"}
    originals = {prep.normalize_text("I think my card is broken"): {"label": "card_not_working", "split": "test", "row": 9}}
    case = prep.convert_tasksource_banking(row, "train", 42, originals)
    assert case["original_split"] == "test"
    assert case["provenance"]["tasksource_split"] == "train"
    assert case["example"]["target"] == {"choice": "card_not_working"}
    assert case["example"]["state"] == {"utterance": "I think my card is broken"}
    bad = {**row, "targets": "card_arrival."}
    with pytest.raises(ValueError, match="gold"):
        prep.convert_tasksource_banking(bad, "train", 42, originals)
    with pytest.raises(ValueError):
        prep.convert_tasksource_banking({**row, "task": "super_glue/boolq"}, "train", 42, originals)


def test_wine_target_is_expert_band_and_not_a_state_feature():
    features = {"fixed acidity": "7.4", "volatile acidity": "0.7", "citric acid": "0", "residual sugar": "1.9", "chlorides": "0.076", "free sulfur dioxide": "11", "total sulfur dioxide": "34", "density": "0.9978", "pH": "3.51", "sulphates": "0.56", "alcohol": "9.4"}
    for quality, expected in [(3, 0), (5, 0), (6, 1), (7, 2), (9, 2)]:
        case = prep.convert_wine({**features, "quality": str(quality)}, "red", quality)
        assert case["example"]["target"] == {"level_index": expected}
        assert len(case["example"]["question"]["criteria"]) == 3
        assert "quality" not in json.dumps(case["example"]["state"])
        assert case["provenance"]["original_quality"] == quality


def test_connected_contexts_cannot_leak_even_across_sources_and_official_heldout_wins():
    a = prep.convert_boolq(BOOLQ, "train", 1)
    b = prep.convert_boolq({**BOOLQ, "question": "is Persian official in Iran"}, "validation", 2)
    c = prep.convert_boolq({**BOOLQ, "passage": "A different passage", "question": "is language mentioned"}, "train", 3)
    c["context_keys"].append(b["context_keys"][0])
    c["provenance"]["dataset"] = "other-copy"
    splits, stats = prep.partition_cases([a, b, c])
    records = [(split, record) for split, rows in splits.items() for record in rows]
    assert len(records) == 3
    assert len({split for split, _ in records}) == 1
    assert records[0][0] in {"validation", "calibration"}
    assert len({r["group_id"] for _, r in records}) == 1
    assert stats["cross_source_context_groups"] == 1


def test_duplicate_rows_drop_once_and_conflicting_gold_drops_whole_duplicate_case():
    a = prep.convert_boolq(BOOLQ, "train", 1)
    b = prep.convert_boolq(BOOLQ, "test", 2)
    splits, stats = prep.partition_cases([a, b])
    assert sum(map(len, splits.values())) == 1
    assert len(splits["test"]) == 1
    assert stats["duplicate_rows_removed"] == 1
    conflicting = prep.convert_boolq({**BOOLQ, "answer": False}, "train", 3)
    splits, stats = prep.partition_cases([a, b, conflicting])
    assert sum(map(len, splits.values())) == 0
    assert stats["conflicting_rows_removed"] == 3


def test_partition_is_repeatable_and_keeps_sentence_permutation_groups_together():
    a = prep.convert_paws(PAWS, "train", 0)
    b = prep.convert_paws({**PAWS, "id": 2, "sentence1": PAWS["sentence2"], "sentence2": PAWS["sentence1"]}, "test", 1)
    first, _ = prep.partition_cases([a, b])
    second, _ = prep.partition_cases([b, a])
    assert first == second
    assert len(first["test"]) == 1


def test_synthetic_rubric_threshold_boundaries_and_heldout_task_family():
    for observed, expected in [(0, 0), (9, 0), (10, 1), (19, 1), (20, 2), (29, 2), (30, 3)]:
        case = prep.synthetic_policy_case("stock", observed, [10, 20, 30], 100)
        assert case["example"]["target"] == {"level_index": expected}
        assert case["provenance"]["label_origin"] == "deterministic-oracle"
    heldout = prep.synthetic_policy_case("latency", 21, [10, 20, 30], 200)
    assert heldout["original_split"] == "test"


def test_manifest_counts_examples_relations_groups_and_hashes(tmp_path):
    a = prep.convert_boolq(BOOLQ, "test", 1)
    b = prep.convert_paws(PAWS, "test", 2)
    splits, stats = prep.partition_cases([a, b])
    manifest = prep.write_corpus(tmp_path, splits, {"fixture": {"license": "test-only"}}, stats)
    assert manifest["splits"]["test"]["bundles"] == 2
    assert manifest["splits"]["test"]["examples"] == 5
    assert manifest["splits"]["test"]["relations"] == {"complement": 2, "invariant": 1}
    assert manifest["splits"]["test"]["groups"] == 2
    assert len(manifest["splits"]["test"]["sha256"]) == 64
    assert len((tmp_path / "test.jsonl").read_text().splitlines()) == 2
    corrupted = copy.deepcopy(splits)
    corrupted["train"] = [copy.deepcopy(splits["test"][0])]
    with pytest.raises(ValueError, match="leak"):
        prep.write_corpus(tmp_path / "bad", corrupted, {}, stats)


def test_bounded_sampling_preserves_minority_labels_and_is_order_independent():
    rows = []
    for index in range(20):
        case = prep.convert_boolq({**BOOLQ, "passage": f"Unique passage {index}", "answer": index < 18}, "train", index)
        rows.append(prep.make_bundle(case, "train", str(index)))
    selected = prep._balanced_take(rows, 4)
    assert sum(row["examples"][0]["target"]["truth"] for row in selected) == 2
    assert prep._balanced_take(list(reversed(rows)), 4) == selected


def test_locked_source_rejects_changed_bytes_and_local_download_restores_missing(tmp_path):
    import hashlib
    source = tmp_path / "source.txt"
    source.write_text("source data\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    lock = {"files": [{"name": "copy.txt", "url": source.as_uri(), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]}
    (raw / "source-lock.json").write_text(json.dumps(lock))
    assert prep.load_locked_sources(raw, download=True) == lock
    (raw / "copy.txt").write_text("changed source")
    with pytest.raises(ValueError, match="checksum"):
        prep.load_locked_sources(raw)
