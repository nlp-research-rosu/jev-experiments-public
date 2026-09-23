import copy
import json
import math

import pytest


def fixture_suite():
    question = {"type": "noul", "instructions": "Does the record confirm completion?"}
    cases = [
        {
            "id": name,
            "family_id": "completion",
            "domain": "test",
            "variant": name,
            "request": {"state": {"result": result}, "questions": {"confirmed": question}},
            "expected": {"confirmed": expected},
            "rationale": {"confirmed": "Evaluator-only explanation"},
        }
        for name, result, expected in [("pending", None, False), ("success", "completed", True)]
    ]
    relation = {
        "id": "completion-flip",
        "kind": "flip",
        "left": {"case_id": "pending", "question_id": "confirmed"},
        "right": {"case_id": "success", "question_id": "confirmed"},
        "reason": "Add confirming evidence",
    }
    return {"cases": cases, "relations": [relation]}


def response(p, model="jev-test-version"):
    return {
        "model": model,
        "answers": {"confirmed": {"type": "noul", "noul": p}},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def test_wire_payload_excludes_expected_answers_and_preserves_question_meaning():
    from experiments.jev_replay import make_payload

    case = fixture_suite()["cases"][0]
    original = copy.deepcopy(case)
    payload = make_payload(case, "jev-latest")
    assert payload == {
        "model": "jev-latest",
        "state": {"result": None},
        "questions": {
            "confirmed": {"type": "noul", "instructions": "Does the record confirm completion?"}
        },
    }
    payload["state"]["result"] = "changed"
    assert case == original


def test_probability_only_loss_does_not_invent_logits_or_hide_zero_gold_probability():
    from experiments.jev_replay import probability_view

    view = probability_view({"false": 1.0, "true": 0.0}, "true")
    assert view["nll"] == pytest.approx(-math.log(1e-12))
    assert view["zero_target_probability"] is True
    assert view["brier"] == 2
    assert view["correct"] is False
    assert "details" not in view
    tied = probability_view({"false": 0.5, "true": 0.5}, "true")
    assert tied["predicted"] is None and tied["ambiguous_top"]


def test_invalid_remote_distributions_and_mismatched_choice_are_rejected():
    from experiments.jev_replay import answer_view

    q = {"type": "choice", "criteria": {"done": "Confirmed", "unknown": "No result"}}
    good = {
        "type": "choice", "choice": "done", "probabilities": {"done": 0.8, "unknown": 0.2},
        "confidence": 0.6,
    }
    assert answer_view(q, "done", good)["correct"]
    for patch in [
        {"probabilities": {"done": 0.8}},
        {"probabilities": {"done": 0.8, "unknown": 0.8}},
        {"probabilities": {"done": float("nan"), "unknown": 0.2}},
        {"probabilities": {"done": True, "unknown": 0}},
        {"choice": "unknown"},
        {"type": "noul"},
    ]:
        with pytest.raises(ValueError):
            answer_view(q, "done", {**good, **patch})


def test_replay_records_raw_responses_and_scores_both_sides_of_a_contrast(tmp_path):
    from experiments.jev_replay import run_suite

    observed = []

    def call(payload):
        observed.append(payload)
        return response(0.2 if payload["state"]["result"] is None else 0.8)

    root = tmp_path / "run"
    result = run_suite(fixture_suite(), root, model="jev-latest", call=call)
    assert result["status"] == "complete"
    assert result["summary"]["overall"]["correct"] == 2
    assert result["summary"]["overall"]["nll"] == pytest.approx(-math.log(0.8))
    assert result["summary"]["relations"]["flip"]["both_correct"] == 1
    assert result["returned_models"] == ["jev-test-version"]
    assert result["usage"] == {"input_tokens": 20, "output_tokens": 4}
    assert len(observed) == 2
    assert all(set(p) == {"state", "questions", "model"} for p in observed)
    raw = [json.loads(line) for line in (root / "responses.jsonl").read_text().splitlines()]
    assert [r["response"]["answers"]["confirmed"]["noul"] for r in raw] == [0.2, 0.8]
    assert all(r["elapsed_ms"] >= 0 for r in raw)
    with pytest.raises(FileExistsError):
        run_suite(fixture_suite(), root, model="jev-latest", call=call)


def test_dry_run_needs_no_key_and_failed_run_keeps_partial_results_without_error_secrets(tmp_path):
    from experiments.jev_replay import run_suite

    dry = run_suite(fixture_suite(), tmp_path / "dry", model="jev-latest", call=None)
    assert dry["status"] == "prepared-not-sent" and dry["requests_completed"] == 0
    calls = []

    def broken(payload):
        calls.append(payload)
        if len(calls) == 2:
            raise RuntimeError("sensitive value must never be logged")
        return response(0.2)

    failed = run_suite(fixture_suite(), tmp_path / "failed", model="jev-latest", call=broken)
    assert failed["status"] == "failed" and failed["requests_completed"] == 1
    assert len(calls) == 2
    assert failed["error_type"] == "RuntimeError"
    assert "sensitive value" not in (tmp_path / "failed" / "report.json").read_text()
    assert len((tmp_path / "failed" / "responses.jsonl").read_text().splitlines()) == 1


def test_model_version_change_fails_run_instead_of_mixing_versions(tmp_path):
    from experiments.jev_replay import run_suite

    result = run_suite(
        fixture_suite(), tmp_path / "versions", model="jev-latest",
        call=lambda payload: response(0.2, "v1" if payload["state"]["result"] is None else "v2"),
    )
    assert result["status"] == "failed"
    assert result["error_type"] == "ValueError"


def test_api_key_is_read_without_shell_evaluation_and_never_echoed(tmp_path, monkeypatch):
    from experiments.jev_replay import load_key

    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    p = tmp_path / "key"
    p.write_text("test-secret\n")
    assert load_key(p) == "test-secret"
    monkeypatch.setenv("TYPESAFE_API_KEY", "environment-secret")
    assert load_key(p) == "environment-secret"
    monkeypatch.delenv("TYPESAFE_API_KEY")
    p.write_text("invalid\nsecond-line")
    with pytest.raises(ValueError) as error:
        load_key(p)
    assert "second-line" not in str(error.value)
    p.write_text("")
    with pytest.raises(ValueError):
        load_key(p)


def test_local_comparison_requires_identical_population_and_labels():
    from experiments.jev_replay import compare_local

    local = [{
        "case_id": "a", "question_id": "q",
        "prediction": {"probabilities": {"false": 0.9, "true": 0.1}, "expected": "false"},
    }]
    remote = [{
        "case_id": "a", "question_id": "q",
        "prediction": {"probabilities": {"false": 0.3, "true": 0.7}, "expected": "false"},
    }]
    result = compare_local(remote, local)
    assert result["questions"] == 1 and result["label_disagreements"] == 1
    assert result["mean_total_variation"] == pytest.approx(0.6)
    assert result["local_correct"] == 1 and result["jev_correct"] == 0
    with pytest.raises(ValueError):
        compare_local(remote, [])
    bad = copy.deepcopy(local)
    bad[0]["prediction"]["expected"] = "true"
    with pytest.raises(ValueError):
        compare_local(remote, bad)


def test_https_transport_sends_only_payload_and_never_follows_redirects(tmp_path, monkeypatch):
    from experiments.jev_replay import APIError, post

    requests, connections = [], []

    class Connection:
        def __init__(self, host, timeout):
            assert host == "api.typesafe.ai" and timeout > 0
            self.closed = False
            connections.append(self)

        def request(self, method, path, body, headers):
            requests.append((method, path, json.loads(body), headers))

        def getresponse(self):
            class Redirect:
                status = 302

                def getheaders(self):
                    return [("Location", "https://untrusted.example/")]

                def read1(self, size):
                    return b""

            return Redirect()

        def close(self):
            self.closed = True

    monkeypatch.setattr("http.client.HTTPSConnection", Connection)
    payload = {"state": "synthetic", "model": "jev-latest", "questions": {}}
    with pytest.raises(APIError) as error:
        post(payload, "test-secret", archive=tmp_path / "archive")
    assert error.value.status == 302
    assert "test-secret" not in str(error.value)
    assert len(requests) == 1 and len(connections) == 1 and connections[0].closed
    method, path, sent, headers = requests[0]
    assert method == "POST" and path == "/v1/systemone" and sent == payload
    assert headers["Authorization"] == "Bearer test-secret"


def test_replay_distinguishes_cached_observations_from_new_usage(tmp_path, monkeypatch):
    from test_jev_archive import transport

    from experiments.jev_archive import ArchiveClient
    from experiments.jev_replay import run_suite

    transport(monkeypatch, [[json.dumps(response(p, "jev-1.13.0")).encode()] for p in (0.2, 0.8)])
    client = ArchiveClient("test-secret", tmp_path / "archive")
    first = run_suite(fixture_suite(), tmp_path / "first", model="jev-1.13.0", call=client)
    assert first["status"] == "complete"
    assert first["cache_hits"] == 0 and first["live_attempts"] == 2
    assert first["new_usage"] == {"input_tokens": 20, "output_tokens": 4}
    cached = run_suite(fixture_suite(), tmp_path / "cached", model="jev-1.13.0", call=ArchiveClient(None, tmp_path / "archive"))
    assert cached["status"] == "complete"
    assert cached["cache_hits"] == 2 and cached["live_attempts"] == 0
    assert cached["new_usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert cached["usage"] == {"input_tokens": 20, "output_tokens": 4}
    assert all(r["cache_hit"] for r in cached["archive_records"])
    rows = [json.loads(line) for line in (tmp_path / "cached" / "responses.jsonl").read_text().splitlines()]
    assert all(row["archive"]["cache_hit"] for row in rows)


def test_interrupted_replay_is_failed_and_links_preserved_partial_archive(tmp_path, monkeypatch):
    from test_jev_archive import transport

    from experiments.jev_archive import ArchiveClient
    from experiments.jev_replay import run_suite

    transport(monkeypatch, [[b"paid partial"]], interrupt=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run_suite(fixture_suite(), tmp_path / "run", model="jev-1.13.0", call=ArchiveClient("test-secret", tmp_path / "archive"))
    report = json.loads((tmp_path / "run" / "report.json").read_text())
    assert report["status"] == "failed" and report["error_type"] == "KeyboardInterrupt"
    assert report["requests_completed"] == 0
    assert report["archive_records"][0]["http_status"] == 200
    assert report["archive_records"][0]["cache_hit"] is False


@pytest.mark.parametrize("raw", [
    {"unknown": 0.05, "no_effect": 0.0, "reversed": 0.01, "effective": 0.93},
    {"unknown": 0.06, "no_effect": 0.01, "reversed": 0.01, "effective": 0.93},
])
def test_choice_rounding_policy_normalizes_cent_grid_and_keeps_reported_values(raw):
    from experiments.jev_replay import answer_view, probability_view

    question = {"type": "choice", "criteria": {k: k for k in raw}}
    answer = {"type": "choice", "choice": "effective", "confidence": 0.91, "probabilities": raw}
    original = copy.deepcopy(answer)
    result = answer_view(question, "effective", answer)
    total = sum(raw.values())
    assert result["probabilities"]["effective"] == pytest.approx(0.93 / total)
    assert result["nll"] == pytest.approx(-math.log(0.93 / total))
    assert result["raw_probabilities"] == raw
    assert result["reported_probability_sum"] == pytest.approx(total)
    assert result["probability_normalized"] is True
    assert result["confidence"] == 0.91
    assert answer == original
    with pytest.raises(ValueError):
        probability_view(raw, "effective")  # Local predictions retain strict policy.


@pytest.mark.parametrize("raw", [
    {"unknown": 0.05, "no_effect": 0.0, "reversed": 0.01, "effective": 0.64},
    {"unknown": 0.051, "no_effect": 0.0, "reversed": 0.01, "effective": 0.929},
    {"unknown": 0.05, "no_effect": 0.0, "reversed": 0.01, "effective": True},
    {"unknown": -0.01, "no_effect": 0.0, "reversed": 0.01, "effective": 0.99},
])
def test_rounding_policy_rejects_large_error_noncent_values_and_invalid_scalars(raw):
    from experiments.jev_replay import answer_view

    question = {"type": "choice", "criteria": {k: k for k in raw}}
    answer = {"type": "choice", "choice": "effective", "confidence": 0.91, "probabilities": raw}
    with pytest.raises(ValueError):
        answer_view(question, "effective", answer)


def test_rounding_policy_still_requires_returned_choice_to_match_raw_maximum():
    from experiments.jev_replay import answer_view

    question = {"type": "choice", "criteria": {"yes": "Yes", "no": "No"}}
    answer = {"type": "choice", "choice": "no", "confidence": 0.5, "probabilities": {"yes": 0.94, "no": 0.05}}
    with pytest.raises(ValueError):
        answer_view(question, "yes", answer)


def test_replay_reports_normalized_vectors_and_policy_without_changing_raw_answers(tmp_path):
    from experiments.jev_replay import run_suite

    suite = fixture_suite()
    for case in suite["cases"]:
        case["request"]["questions"]["confirmed"] = {
            "type": "choice", "criteria": {"yes": "Confirmed", "no": "Not confirmed"},
        }
        case["expected"]["confirmed"] = "yes" if case["id"] == "success" else "no"

    def call(request):
        raw = {"yes": 0.94, "no": 0.05} if request["state"]["result"] is None else {"yes": 0.95, "no": 0.05}
        return {
            "model": "jev-1.13.0", "usage": {"input_tokens": 10, "output_tokens": 2},
            "answers": {"confirmed": {"type": "choice", "choice": "yes", "confidence": 0.91, "probabilities": raw}},
        }

    report = run_suite(suite, tmp_path / "run", model="jev-1.13.0", call=call)
    assert report["status"] == "complete"
    assert report["normalized_probability_vectors"] == 1
    assert report["metric_policy"]["validation_policy_version"] == 3
    raw = json.loads((tmp_path / "run" / "responses.jsonl").read_text().splitlines()[0])
    assert raw["response"]["answers"]["confirmed"]["probabilities"] == {"yes": 0.94, "no": 0.05}


def score_fixture():
    question = {"type": "score", "instructions": "How complete is the work?", "criteria": [
        "No verified progress", {"level": "Partial progress"}, ["All steps verified"],
    ]}
    answer = {
        "type": "score", "score": 1.7, "confidence": 0.83,
        "legend": {"0": "No verified progress", "1": {"level": "Partial progress"}, "2": ["All steps verified"]},
        "probabilities": {"0": 0.1, "1": 0.1, "2": 0.8},
    }
    return question, answer


def test_score_uses_full_distribution_preserves_reported_values_and_never_invents_logits():
    from experiments.jev_replay import answer_view

    question, answer = score_fixture()
    original = copy.deepcopy(answer)
    result = answer_view(question, 2, answer)
    assert result["expected"] == "2" and result["predicted"] == "2"
    assert result["nll"] == pytest.approx(-math.log(0.8))
    assert result["brier"] == pytest.approx(0.06)
    assert result["score_mean"] == pytest.approx(1.7)
    assert result["score_absolute_error"] == pytest.approx(0.3)
    assert result["returned_score"] == 1.7 and result["confidence"] == 0.83
    assert "details" not in result and "calibrated" not in result
    assert answer == original


def test_score_accepts_cent_rounding_but_keeps_raw_vector_and_returned_mean():
    from experiments.jev_replay import answer_view

    question, answer = score_fixture()
    answer.update(score=1.32, probabilities={"0": 0.33, "1": 0.01, "2": 0.65})
    result = answer_view(question, 2, answer)
    assert result["score_mean"] == pytest.approx(1.31 / 0.99)
    assert result["returned_score"] == 1.32
    assert result["raw_probabilities"] == answer["probabilities"]
    assert result["reported_probability_sum"] == pytest.approx(0.99)
    assert result["probability_normalized"] is True
    assert result["confidence"] == 0.83


@pytest.mark.parametrize("patch", [
    {"probabilities": {"0": 0.2, "2": 0.8}},
    {"probabilities": {"0": 0.1, "1": 0.1, "2": 0.7, "3": 0.1}},
    {"probabilities": {"0": 0.1, "1": 0.1, "2": 0.7}},
    {"probabilities": {"0": 0.105, "1": 0.1, "2": 0.785}},
    {"probabilities": {"0": -0.1, "1": 0.1, "2": 1.0}},
    {"probabilities": {"0": False, "1": 0.2, "2": 0.8}},
    {"probabilities": {"0": float("nan"), "1": 0.2, "2": 0.8}},
    {"score": -0.01}, {"score": 2.01}, {"score": float("inf")}, {"score": True},
    {"score": None}, {"score": 0.7}, {"confidence": None}, {"confidence": 1.01},
    {"legend": {"0": "Different rubric", "1": {"level": "Partial progress"}, "2": ["All steps verified"]}},
    {"legend": {"0": "No verified progress", "2": ["All steps verified"]}},
])
def test_score_rejects_invalid_full_vectors_means_confidence_and_legends(patch):
    from experiments.jev_replay import answer_view

    question, answer = score_fixture()
    with pytest.raises(ValueError):
        answer_view(question, 2, {**answer, **patch})


@pytest.mark.parametrize("expected", [-1, 3, "2", True, None, 1.5])
def test_score_target_must_be_a_declared_integer_index(expected):
    from experiments.jev_replay import answer_view

    question, answer = score_fixture()
    with pytest.raises(ValueError):
        answer_view(question, expected, answer)


def test_score_mean_rounding_tolerance_covers_cent_bins_without_accepting_unrelated_mean():
    from experiments.jev_replay import answer_view

    question, answer = score_fixture()
    # These bins can arise from (0.334, 0.334, 0.332), whose mean rounds to 1.00.
    answer.update(score=1.0, probabilities={"0": 0.33, "1": 0.33, "2": 0.33})
    assert answer_view(question, 1, answer)["returned_score"] == 1.0
    answer.update(score=1.0, probabilities={"0": 0.12345, "1": 0.34567, "2": 0.53088})
    with pytest.raises(ValueError):
        answer_view(question, 1, answer)
    answer["score"] = 1.41  # Rounded from the exact vector mean 1.40743.
    assert answer_view(question, 1, answer)["returned_score"] == 1.41


def test_score_replay_records_probabilistic_mean_error_and_raw_response(tmp_path):
    from experiments.jev_replay import run_suite

    question, answer = score_fixture()
    suite = {"cases": [{
        "id": "score", "family_id": "completion", "domain": "test", "variant": "score", "layout": "individual",
        "request": {"state": {"result": "complete"}, "questions": {"rating": question}},
        "expected": {"rating": 2}, "rationale": {"rating": "All steps are verified"},
    }], "relations": []}
    raw = {"model": "jev-1.13.0", "answers": {"rating": answer}, "usage": {"input_tokens": 10, "output_tokens": 2}}
    report = run_suite(suite, tmp_path / "score", model="jev-1.13.0", call=lambda request: raw)
    assert report["status"] == "complete"
    assert report["summary"]["by_primitive"]["score"]["score_mean_absolute_error"] == pytest.approx(0.3)
    saved = json.loads((tmp_path / "score" / "responses.jsonl").read_text())
    assert saved["response"] == raw
    predictions = json.loads((tmp_path / "score" / "predictions.json").read_text())
    assert predictions[0]["layout"] == "individual"


@pytest.mark.parametrize("criteria", [None, {}, [], "Unordered rubric", [False, "Complete"]])
def test_score_rejects_missing_or_invalid_ordered_rubric(criteria):
    from experiments.jev_replay import answer_view

    question, answer = score_fixture()
    question["criteria"] = criteria
    with pytest.raises(ValueError):
        answer_view(question, 0, answer)


def test_score_accepts_sdk_singleton_rubric_without_dividing_by_zero():
    from experiments.jev_replay import answer_view

    question = {"type": "score", "criteria": ["The only level"]}
    answer = {
        "type": "score", "score": 0, "confidence": 0.73,
        "legend": {"0": "The only level"}, "probabilities": {"0": 1},
    }
    result = answer_view(question, 0, answer)
    assert result["score_mean"] == result["score_absolute_error"] == result["score_normalized_absolute_error"] == 0
