"""Offline collector contracts; real archive writes, only HTTPS/time are replaced."""

import copy
import http.client
import json
import threading
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path

import pytest


class Clock:
    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()

    def monotonic(self):
        with self.lock:
            return self.now

    def sleep(self, seconds):
        with self.lock:
            self.now += seconds

    def time(self):
        return 1_800_000_000 + self.monotonic()


def suite_file(tmp_path, count=1, duplicate=False):
    cases = []
    for i in range(count):
        cases.append({
            "id": f"case-{i}", "family_id": "family", "domain": "test", "variant": str(i),
            "request": {"state": {"result": "done", "index": 0 if duplicate else i},
                        "questions": {"confirmed": {"type": "noul", "instructions": "Completed?"}}},
            "expected": {"confirmed": True}, "rationale": {"confirmed": "PRIVATE_GOLD"},
        })
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"cases": cases, "relations": []}))
    return path


def response(p=0.8):
    return {"model": "jev-1.13.0", "answers": {"confirmed": {"type": "noul", "noul": p}},
            "usage": {"input_tokens": 100, "output_tokens": 2}}


def transport(monkeypatch, replies, clock=None):
    observed = []
    guard = threading.Lock()

    class Connection:
        def __init__(self, host, timeout):
            assert host == "api.typesafe.ai"

        def request(self, method, path, body, headers):
            with guard:
                observed.append({"payload": json.loads(body), "time": clock.monotonic() if clock else 0})
                assert replies, "Unexpected additional paid attempt"
                self.reply = replies.pop(0)

        def getresponse(self):
            reply = self.reply
            if isinstance(reply, BaseException):
                raise reply

            class Remote:
                status = reply.get("status", 200)
                length = None

                def getheaders(self):
                    return reply.get("headers", [])

                def read1(self, size):
                    if hasattr(self, "done"):
                        return b""
                    self.done = True
                    return reply.get("body", json.dumps(response()).encode())

            return Remote()

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", Connection)
    return observed


def run(tmp_path, monkeypatch, replies, *, count=1, duplicate=False, **kwargs):
    from experiments.calibrated_jev import collect

    clock = Clock()
    sent = transport(monkeypatch, replies, clock)
    suite = suite_file(tmp_path, count, duplicate)
    output = tmp_path / "out"
    report = collect(suite, output, run=True, key="test-secret", workers=1,
                     clock=clock, random_value=lambda: 0.5, lock_path=tmp_path / "owner.lock", **kwargs)
    return report, sent, suite, output


def test_limiter_smooth_ramp_no_accumulated_burst_and_token_pressure():
    from experiments.calibrated_jev import GlobalLimiter

    clock = Clock()
    limiter = GlobalLimiter(clock=clock)
    starts = []
    for _ in range(22):
        limiter.acquire(100)
        starts.append(clock.monotonic())
        limiter.success(100, 100)
    assert starts[:20] == pytest.approx([i * 0.2 for i in range(20)])
    assert starts[-1] - starts[-2] == pytest.approx(0.1)
    clock.sleep(100)
    limiter.acquire(300_000)
    first = clock.monotonic()
    limiter.acquire(100)
    assert clock.monotonic() - first == pytest.approx(3.0)
    limiter.success(100, 200_000)
    limiter.acquire(100)
    first = clock.monotonic()
    limiter.acquire(100)
    assert clock.monotonic() - first == pytest.approx(2.0)


def test_throttle_slows_global_limiter_and_never_caps_server_wait():
    from experiments.calibrated_jev import GlobalLimiter, retry_after_seconds

    clock = Clock()
    limiter = GlobalLimiter(clock=clock)
    limiter.acquire(100)
    limiter.throttle(123)
    limiter.acquire(100)
    assert clock.monotonic() >= 123
    previous = clock.monotonic()
    limiter.acquire(100)
    assert clock.monotonic() - previous == pytest.approx(0.4)
    future = format_datetime(datetime.fromtimestamp(clock.time() + 190, timezone.utc), usegmt=True)
    # HTTP dates have whole-second precision: this clock is at *.4 seconds.
    assert retry_after_seconds(future, clock.time()) == pytest.approx(189.6)
    assert retry_after_seconds("999", clock.time()) == 999
    assert retry_after_seconds("bad header", clock.time()) is None


def test_dryrun_emits_safe_requests_and_never_loads_key_or_connects(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    suite = suite_file(tmp_path, 2, duplicate=True)
    transport(monkeypatch, [])
    report = collect(suite, tmp_path / "out", lock_path=tmp_path / "owner.lock")
    assert report["status"] == "prepared-not-sent"
    assert report["unique_requests"] == 1
    assert report["cases_planned"] == 2
    assert "PRIVATE_GOLD" not in (tmp_path / "out" / "requests.jsonl").read_text()
    assert not (tmp_path / "out" / "targets.json").exists()


def test_dedup_durable_targets_and_resume_validate_without_second_call(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    report, sent, suite, output = run(tmp_path, monkeypatch, [{}], count=2, duplicate=True)
    assert report["status"] == "complete"
    assert len(sent) == 1
    assert set(sent[0]["payload"]) == {"model", "state", "questions"}
    targets = json.loads((output / "targets.json").read_text())
    assert {row["case_id"] for row in targets["rows"]} == {"case-0", "case-1"}
    assert all(row["probabilities"] == pytest.approx({"false": .2, "true": .8}) for row in targets["rows"])
    observations = {row["observation_path"] for row in targets["rows"]}
    assert len(observations) == 1
    assert Path(observations.pop()).name == "capture.json"
    original = {p: p.read_bytes() for p in (output / "archive").rglob("*") if p.is_file()}
    resumed = collect(suite, output, run=True, key=None, workers=1, lock_path=tmp_path / "owner.lock")
    assert resumed["status"] == "complete"
    assert len(sent) == 1
    assert all(p.read_bytes() == data for p, data in original.items())
    assert report["new_usage"] == {"input_tokens": 100, "output_tokens": 2}


def test_retry_after_and_five_retry_ceiling_preserve_every_attempt(tmp_path, monkeypatch):
    replies = [{"status": 429, "headers": [("Retry-After", "120")], "body": b'{"error":"busy"}'}] * 6
    report, sent, _, output = run(tmp_path, monkeypatch, replies)
    assert report["status"] == "failed"
    assert len(sent) == 6
    assert all(b["time"] - a["time"] >= 120 for a, b in zip(sent, sent[1:]))
    assert report["retries"] == 5
    assert report["attempts"] == 6
    assert len(list((output / "archive").glob("*/*/capture.json"))) == 6
    assert not (output / "targets.json").exists()


def test_retry_backoff_and_success_keep_error_body_and_usage(tmp_path, monkeypatch):
    replies = [{"status": 529, "body": b'{"error":"test-secret busy"}'},
               {"status": 429, "body": b'{"error":"busy"}'}, {}]
    report, sent, _, output = run(tmp_path, monkeypatch, replies)
    assert report["status"] == "complete"
    assert sent[1]["time"] - sent[0]["time"] >= 2
    assert sent[2]["time"] - sent[1]["time"] >= 4
    assert report["attempts"] == 3 and report["retries"] == 2
    assert report["unknown_usage_attempts"] == 2
    assert all(b"test-secret" not in p.read_bytes() for p in output.rglob("*") if p.is_file())


@pytest.mark.parametrize("reply", [
    {"status": 401, "body": b'{"error":"bad auth"}'},
    {"body": b'{"invalid":"schema"}'},
    TimeoutError("ambiguous timeout"),
])
def test_terminal_errors_are_not_retried_even_on_resume(tmp_path, monkeypatch, reply):
    from experiments.calibrated_jev import collect

    report, sent, suite, output = run(tmp_path, monkeypatch, [reply])
    assert report["status"] == "failed"
    assert len(sent) == 1
    again = collect(suite, output, run=True, key="test-secret", workers=1, lock_path=tmp_path / "owner.lock")
    assert again["status"] == "failed"
    assert len(sent) == 1
    assert len(list((output / "archive").glob("*/*/capture.json"))) == 1


def test_changed_suite_and_corrupt_completed_body_fail_before_any_new_call(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    _, sent, suite, output = run(tmp_path, monkeypatch, [{}])
    original = suite.read_bytes()
    suite.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="fingerprint|manifest|suite"):
        collect(suite, output, run=True, key="test-secret", lock_path=tmp_path / "owner.lock")
    suite.write_bytes(original)
    body = next((output / "archive").glob("*/*/response.body"))
    body.write_bytes(json.dumps(response(.1)).encode())
    with pytest.raises(ValueError, match="archive|observation|completed"):
        collect(suite, output, run=True, key="test-secret", lock_path=tmp_path / "owner.lock")
    assert len(sent) == 1


def test_one_owner_lock_rejects_concurrent_collector(tmp_path):
    from experiments.calibrated_jev import collector_lock

    with collector_lock(tmp_path / "owner.lock"):
        with pytest.raises(RuntimeError, match="collector|owner|lock"):
            with collector_lock(tmp_path / "owner.lock"):
                pytest.fail("Acquired an already owned lock")


def test_worker_limit_cannot_exceed_ten(tmp_path):
    from experiments.calibrated_jev import collect

    with pytest.raises(ValueError, match="workers|10"):
        collect(suite_file(tmp_path), tmp_path / "out", workers=11)


def test_progress_counts_waiting_workers_as_queued_not_http_in_flight(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    entered, release = threading.Event(), threading.Event()
    clock = Clock()
    original_sleep = clock.sleep

    def pause_after_first_start(seconds):
        entered.set()
        assert release.wait(5)
        original_sleep(seconds)

    clock.sleep = pause_after_first_start
    suite = suite_file(tmp_path, 3)
    transport(monkeypatch, [{}] * 3, clock)
    output = tmp_path / "out"
    result = {}

    def collect_in_thread():
        try:
            result["report"] = collect(suite, output, run=True, key="test-secret", workers=3,
                                       clock=clock, lock_path=tmp_path / "owner.lock")
        except BaseException as error:
            result["error"] = error

    thread = threading.Thread(target=collect_in_thread)
    thread.start()
    try:
        assert entered.wait(5)
        # The first response is allowed to finish before examining waiting work.
        for _ in range(1000):
            progress = json.loads((output / "report.json").read_text())
            if progress["validated_unique"] == 1:
                break
            threading.Event().wait(.001)
        assert progress["validated_unique"] == 1
        assert progress["in_flight"] == 0
        assert progress["queued"] == 2
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive()
    assert "error" not in result
    assert result["report"]["status"] == "complete"


def test_shared_archive_respects_each_suite_and_reuses_exact_observation(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    report, sent, suite, output = run(tmp_path, monkeypatch, [{}])
    original = json.loads((output / "targets.json").read_text())["rows"][0]
    second = collect(suite, tmp_path / "second", run=True, archive=output / "archive",
                     lock_path=tmp_path / "owner.lock")
    repeated = json.loads((tmp_path / "second/targets.json").read_text())["rows"][0]
    assert second["status"] == report["status"] == "complete"
    assert repeated["observation_path"] == original["observation_path"]
    assert len(sent) == 1 and second["attempts"] == 0


def test_interrupted_intent_without_response_stops_before_new_paid_attempt(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    suite = suite_file(tmp_path)
    output = tmp_path / "out"
    collect(suite, output, lock_path=tmp_path / "owner.lock")
    digest = next(iter(json.loads((output / "manifest.json").read_text())["requests"]))
    (output / "attempts/interrupted.start.json").write_text(json.dumps({
        "attempt_id": "interrupted", "request_sha256": digest, "model": "jev-1.13.0",
        "retry_index": 0, "started_utc": "2026-09-22T00:00:00Z", "estimated_input_tokens": 500,
    }))
    sent = transport(monkeypatch, [])
    report = collect(suite, output, run=True, key="test-secret", lock_path=tmp_path / "owner.lock")
    assert report["status"] == "failed"
    assert any("Ambiguous" in reason for reason in report["failures"].values())
    assert not sent


def test_changed_response_policy_fingerprint_prevents_resume(tmp_path, monkeypatch):
    from experiments import calibrated_jev

    suite = suite_file(tmp_path)
    output = tmp_path / "out"
    calibrated_jev.collect(suite, output, lock_path=tmp_path / "owner.lock")
    changed = copy.deepcopy(calibrated_jev.PROTOCOL)
    changed["archive_validation_policy_version"] += 1
    monkeypatch.setattr(calibrated_jev, "PROTOCOL", changed)
    with pytest.raises(ValueError, match="fingerprint"):
        calibrated_jev.collect(suite, output, run=True, key="test-secret", lock_path=tmp_path / "owner.lock")


def test_ten_concurrent_responses_keep_each_workers_archive_identity(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    full, release, guard = threading.Event(), threading.Event(), threading.Lock()
    state = {"active": 0, "maximum": 0, "sent": 0}

    class Connection:
        def __init__(self, host, timeout):
            pass

        def request(self, method, path, body, headers):
            self.index = json.loads(body)["state"]["index"]
            with guard:
                state["active"] += 1
                state["sent"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
                if state["active"] == 10:
                    full.set()

        def getresponse(self):
            index = self.index

            class Remote:
                status, length = 200, None

                def getheaders(self):
                    return []

                def read1(self, size):
                    if hasattr(self, "done"):
                        return b""
                    assert release.wait(10)
                    self.done = True
                    return json.dumps(response((index + 1) / 13)).encode()

            return Remote()

        def close(self):
            with guard:
                state["active"] -= 1

    monkeypatch.setattr(http.client, "HTTPSConnection", Connection)
    suite, output, result = suite_file(tmp_path, 12), tmp_path / "out", {}

    def collect_in_thread():
        try:
            result["report"] = collect(suite, output, run=True, key="test-secret", clock=Clock(),
                                       lock_path=tmp_path / "owner.lock")
        except BaseException as error:
            result["error"] = error

    thread = threading.Thread(target=collect_in_thread)
    thread.start()
    try:
        assert full.wait(10)
        assert state["sent"] == 10
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and "error" not in result
    assert result["report"]["status"] == "complete"
    assert state["maximum"] == 10 and state["sent"] == 12
    targets = json.loads((output / "targets.json").read_text())["rows"]
    assert len({row["observation_path"] for row in targets}) == 12
    for row in targets:
        index = int(row["case_id"].split("-")[1])
        assert row["probabilities"]["true"] == pytest.approx((index + 1) / 13)


def test_choice_and_score_use_existing_rounding_policy_and_full_string_labels(tmp_path, monkeypatch):
    from experiments.calibrated_jev import collect

    suite = suite_file(tmp_path)
    fixture = json.loads(suite.read_text())
    case = fixture["cases"][0]
    case["request"]["questions"].update({
        "category": {"type": "choice", "instructions": "Category?", "criteria": {"a": "A", "b": "B", "c": "C"}},
        "rating": {"type": "score", "instructions": "Rating?", "criteria": ["low", "medium", "high"]},
    })
    case["expected"].update(category="a", rating=1)
    case["rationale"].update(category="PRIVATE", rating="PRIVATE")
    suite.write_text(json.dumps(fixture))
    remote = response()
    remote["answers"].update({
        "category": {"type": "choice", "choice": "a", "confidence": .33,
                     "probabilities": {"a": .33, "b": .33, "c": .33}},
        "rating": {"type": "score", "score": 1.0, "confidence": .33,
                   "legend": {"0": "low", "1": "medium", "2": "high"},
                   "probabilities": {"0": .33, "1": .33, "2": .33}},
    })
    raw = json.dumps(remote).encode()
    transport(monkeypatch, [{"body": raw}])
    output = tmp_path / "out"
    report = collect(suite, output, run=True, key="test-secret", lock_path=tmp_path / "owner.lock")
    assert report["status"] == "complete"
    rows = {row["question_id"]: row for row in json.loads((output / "targets.json").read_text())["rows"]}
    assert rows["category"]["probabilities"] == pytest.approx({"a": 1/3, "b": 1/3, "c": 1/3})
    assert rows["rating"]["probabilities"] == pytest.approx({"0": 1/3, "1": 1/3, "2": 1/3})
    assert Path(rows["rating"]["observation_path"]).with_name("response.body").read_bytes() == raw
