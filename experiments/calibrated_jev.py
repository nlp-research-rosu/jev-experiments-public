"""Bounded, resumable Jev target collection with durable, partition-local archives.

Default invocation prepares requests offline. Only --run permits paid requests.
Failures preserve attempts; terminal/ambiguous failures require explicit review.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import random
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from experiments.jev_archive import (
    APIError, ArchiveClient, DEFAULT_MODEL, VALIDATION_POLICY_VERSION,
    _body, _fixed, _parse, _sync_directory, _write_once,
)
from experiments.jev_replay import answer_view, load_key, make_payload
from experiments.revised_metrics import validate_suite

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK = ROOT / "reports/calibrated-screen-v1/jev-collector.lock"
MAX_RETRIES = 5
PROTOCOL = {
    "format_version": 1, "target_schema_version": 1,
    "archive_validation_policy_version": VALIDATION_POLICY_VERSION,
    "initial_requests_per_second": 5, "maximum_requests_per_second": 10,
    "healthy_fresh_batch": 20, "maximum_in_flight": 10,
    "estimated_input_tokens_per_second": 100_000,
    "estimated_tokens": "UTF-8 serialized payload bytes + 256 + 64/question; upward-only observed usage ratio",
    "maximum_retries": MAX_RETRIES, "retry_statuses": [429, 529],
    "backoff": "2*2^retry capped at 60 seconds, plus uniform [0,1] seconds; server Retry-After is never capped",
    "cache_scope": "explicit partition archive; exact request and fixed returned model; earliest valid observation",
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _request_hash(request):
    return _sha(json.dumps(request, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False).encode())


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _replace_json(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(_body(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def collector_lock(path=DEFAULT_LOCK):
    """One process owns the shared allowance, including across output directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another Jev collector owns the collection lock") from None
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(json.dumps({"pid": os.getpid(), "started_utc": _utc()}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def retry_after_seconds(value, now):
    """Parse seconds or HTTP-date without reducing a server's requested wait."""
    if not isinstance(value, str):
        return None
    try:
        delay = float(value)
        if math.isfinite(delay) and delay >= 0:
            return delay
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(value)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return max(0.0, moment.timestamp() - now)
    except (TypeError, ValueError, OverflowError):
        return None


class GlobalLimiter:
    """Smooth shared attempt pacing; sleeping never holds the feedback lock."""
    def __init__(self, *, clock=time):
        self.clock = clock
        self.lock = threading.Lock()
        self.rate = 5.0
        self.healthy = 0
        self.token_ratio = 1.0
        self.next_start = 0.0
        self.cooldown_until = 0.0

    def acquire(self, estimated_tokens, stop=None):
        while True:
            if stop is not None and stop.is_set():
                return False
            with self.lock:
                now = self.clock.monotonic()
                delay = max(self.next_start, self.cooldown_until) - now
                if delay <= 1e-10:
                    self.next_start = now + max(1 / self.rate, estimated_tokens * self.token_ratio / 100_000)
                    return True
            self.clock.sleep(min(delay, 0.25))

    def success(self, estimated_tokens, reported_tokens):
        with self.lock:
            self.token_ratio = max(self.token_ratio, reported_tokens / max(estimated_tokens, 1))
            self.healthy += 1
            if self.healthy >= 20:
                self.rate = min(10.0, self.rate * 2)
                self.healthy = 0

    def throttle(self, delay):
        with self.lock:
            self.rate = max(0.25, self.rate / 2)
            self.healthy = 0
            self.cooldown_until = max(self.cooldown_until, self.clock.monotonic() + delay)

    def snapshot(self):
        with self.lock:
            return {"requests_per_second": self.rate, "estimated_token_multiplier": self.token_ratio,
                    "cooldown_seconds_remaining": max(0, self.cooldown_until - self.clock.monotonic())}


def _estimate(payload):
    return len(_body(payload)) + 256 + 64 * len(payload["questions"])


def _usage(value):
    if isinstance(value, dict) and all(type(value.get(k)) is int and value[k] >= 0
                                       for k in ("input_tokens", "output_tokens")):
        return {k: value[k] for k in ("input_tokens", "output_tokens")}
    return None


def _cached(payload, directory):
    # A distinct client per call avoids last_record races. This cache-only method
    # cannot send a request; the paid client below is deliberately fresh=True.
    client = ArchiveClient(None, directory)
    response = client._cached(payload)
    return response, copy.deepcopy(client.last_record)


def _rows(cases, response, observation):
    result = []
    for case in cases:
        for qid, question in case["request"]["questions"].items():
            # A dummy member of the answer space invokes the established policy
            # without reading gold labels. Only its probability vector is used.
            dummy = False if question["type"] == "noul" else 0 if question["type"] == "score" else next(iter(question["criteria"]))
            view = answer_view(question, dummy, response["answers"][qid])
            result.append({"case_id": case["id"], "question_id": qid,
                           "probabilities": view["probabilities"],
                           "request_sha256": _request_hash(case["request"]),
                           "observation_path": observation})
    return result


def collect(suite_path, output, *, run=False, model=DEFAULT_MODEL, workers=10,
            archive=None, key=None, lock_path=DEFAULT_LOCK, clock=time, random_value=random.random):
    """Prepare or collect one immutable suite. Return a durable progress report.

    `archive` must be a deliberately shared, permitted partition archive. Never
    point this collector at quarantined evaluations. The default is output/archive.
    """
    if type(workers) is not int or not 1 <= workers <= 10:
        raise ValueError("workers must be between 1 and 10")
    if not _fixed(model):
        raise ValueError("Collector requires a pinned model such as jev-1.13.0")
    suite_path, output = Path(suite_path).resolve(), Path(output).resolve()
    archive = Path(archive).resolve() if archive else output / "archive"
    suite = _parse(suite_path.read_bytes())
    validate_suite(suite)
    grouped = {}
    for case in suite["cases"]:
        grouped.setdefault(_request_hash(case["request"]), []).append(case)
    manifest = {"protocol": PROTOCOL, "source_suite_sha256": _sha(suite_path.read_bytes()),
                "model": model, "archive": str(archive),
                "requests": {digest: [case["id"] for case in cases] for digest, cases in grouped.items()}}
    manifest["schema_fingerprint"] = _sha(_body(PROTOCOL))
    with collector_lock(lock_path):
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if _parse(manifest_path.read_bytes()) != manifest:
                raise ValueError("Collector manifest/suite/schema fingerprint changed; preserve this output")
        else:
            if any(output.iterdir()):
                raise ValueError("Output lacks a matching manifest; refusing unsafe resume")
            _write_once(manifest_path, manifest)
        planned = b"".join(_body({"case_ids": [case["id"] for case in cases], "request_sha256": digest,
                                  "request": make_payload(cases[0], model)}) + b"\n"
                           for digest, cases in grouped.items())
        requests_path = output / "requests.jsonl"
        if requests_path.exists():
            if requests_path.read_bytes() != planned:
                raise ValueError("Saved request manifest changed")
        else:
            _write_once(requests_path, planned)
        return _collect_locked(suite_path, output, grouped, manifest, run=run, model=model,
                               workers=workers, archive=archive, key=key, clock=clock, random_value=random_value)


def _collect_locked(suite_path, output, grouped, manifest, *, run, model, workers, archive, key, clock, random_value):
    started = clock.monotonic()
    limiter = GlobalLimiter(clock=clock)
    stop = threading.Event()
    mutex = threading.RLock()
    completed_dir, attempts_dir = output / "completed", output / "attempts"
    completed_dir.mkdir(exist_ok=True)
    attempts_dir.mkdir(exist_ok=True)
    responses, observations, attempts, failures, cached_digests = {}, {}, {}, {}, set()
    in_flight, retrying = set(), set()
    phase = "prepared-not-sent"

    # Intent precedes network I/O. Unmatched intent is ambiguous unless a complete
    # validated archive observation recovers it; it must never trigger a new call.
    for path in sorted(attempts_dir.glob("*.start.json")):
        intent = _parse(path.read_bytes())
        digest = intent["request_sha256"]
        if digest not in grouped or intent["model"] != model:
            raise ValueError("Attempt ledger identity differs from manifest")
        result_path = path.with_name(path.name.replace(".start.json", ".result.json"))
        result = _parse(result_path.read_bytes()) if result_path.exists() else None
        if result is not None and result["attempt_id"] != intent["attempt_id"]:
            raise ValueError("Attempt ledger result identity mismatch")
        attempts.setdefault(digest, []).append({"intent": intent, "result": result})
    for digest, cases in grouped.items():
        payload = make_payload(cases[0], model)
        response, metadata = _cached(payload, archive / digest)
        completed_path = completed_dir / (digest + ".json")
        if completed_path.exists():
            saved = _parse(completed_path.read_bytes())
            if (saved.get("request_sha256") != digest or saved.get("model") != model
                    or response is None or saved.get("observation_path") != metadata["archive_record"]
                    or saved.get("capture_sha256") != _sha(Path(metadata["archive_record"]).read_bytes())):
                raise ValueError("Completed archive observation failed validation; no replacement request was sent")
        if response is not None:
            responses[digest], observations[digest] = response, metadata["archive_record"]
            cached_digests.add(digest)
            if not completed_path.exists() and run:
                _write_once(completed_path, {"request_sha256": digest, "model": model,
                            "observation_path": metadata["archive_record"],
                            "capture_sha256": _sha(Path(metadata["archive_record"]).read_bytes())})
            continue
        previous = attempts.get(digest, [])
        if previous:
            if any(item["result"] is None for item in previous):
                failures[digest] = "Ambiguous interrupted attempt; retained for manual review"
            elif any(item["result"]["status"] != "retryable" for item in previous):
                failures[digest] = "Prior terminal error; retained for manual review"
            elif len(previous) >= MAX_RETRIES + 1:
                failures[digest] = "Retry limit already exhausted"
            else:
                for item in previous:
                    limiter.throttle(item["result"]["retry_delay_seconds"])
        elif (archive / digest).exists() and any((archive / digest).iterdir()):
            failures[digest] = "Unvalidated archived attempt without ledger; retained for manual review"

    def report():
        with mutex:
            entries = [entry for group in attempts.values() for entry in group]
            captured = [entry["result"] for entry in entries if entry["result"] is not None
                        and entry["result"].get("observation_path")]
            usages = [entry["usage"] for entry in captured if entry.get("usage") is not None]
            new_usage = {k: sum(u[k] for u in usages) for k in ("input_tokens", "output_tokens")}
            usage = {k: sum(response["usage"][k] for response in responses.values())
                     for k in ("input_tokens", "output_tokens")}
            value = {
                "status": phase, "pid": os.getpid(), "updated_utc": _utc(),
                "elapsed_seconds_this_session": clock.monotonic() - started,
                "model": model, "source_suite_sha256": manifest["source_suite_sha256"],
                "schema_fingerprint": manifest["schema_fingerprint"], "protocol": PROTOCOL,
                "cases_planned": sum(len(cases) for cases in grouped.values()),
                "unique_requests": len(grouped), "duplicate_cases_avoided": sum(len(v) - 1 for v in grouped.values()),
                "queued": len(set(grouped) - set(responses) - set(failures) - in_flight - retrying),
                "in_flight": len(in_flight), "retrying": len(retrying),
                "validated_unique": len(responses), "captured": len(captured),
                "completed_cases": sum(len(grouped[digest]) for digest in responses),
                "failed": len(failures), "failures": dict(failures),
                "attempts": len(entries), "retries": sum(max(0, len(v) - 1) for v in attempts.values()),
                "cache_hits_this_session": len(cached_digests), "usage": usage, "new_usage": new_usage,
                "unknown_usage_attempts": len(entries) - len(usages),
                "estimated_attempt_input_tokens": sum(entry["intent"]["estimated_input_tokens"] for entry in entries),
                "reported_input_token_cost_estimate_usd": new_usage["input_tokens"] * 0.042 / 1_000_000,
                "usage_note": "Reported usage and list-price estimate are not verified billing or balance. Missing usage remains unknown. UTF-8 estimates cannot guarantee provider token limits.",
                "raw_observations": sorted({v["observation_path"] for v in captured} | set(observations.values())),
                "limiter": limiter.snapshot(),
            }
            _replace_json(output / "report.json", value)
            return value

    def complete(digest, response, record):
        observation = record["archive_record"]
        _write_once(completed_dir / (digest + ".json"), {
            "request_sha256": digest, "model": model, "observation_path": observation,
            "capture_sha256": _sha(Path(observation).read_bytes()),
        })
        with mutex:
            responses[digest], observations[digest] = response, observation

    def work(digest):
        payload = make_payload(grouped[digest][0], model)
        client = ArchiveClient(key, archive / digest, fresh=True)
        estimated = _estimate(payload)
        try:
            while True:
                if not limiter.acquire(estimated, stop):
                    return
                with mutex:
                    retrying.discard(digest)
                    in_flight.add(digest)
                    index = len(attempts.get(digest, []))
                    attempt_id = uuid.uuid4().hex
                    intent = {"attempt_id": attempt_id, "request_sha256": digest, "model": model,
                              "retry_index": index, "started_utc": _utc(), "estimated_input_tokens": estimated}
                    _write_once(attempts_dir / (attempt_id + ".start.json"), intent)
                    entry = {"intent": intent, "result": None}
                    attempts.setdefault(digest, []).append(entry)
                    report()
                error, response = None, None
                try:
                    response = client(payload)
                except BaseException as caught:
                    error = caught
                metadata = copy.deepcopy(client.last_record)
                capture, validation = {}, {}
                if metadata is not None:
                    capture = _parse(Path(metadata["archive_record"]).read_bytes())
                    validation_path = Path(metadata["archive_record"]).with_name("validation.json")
                    if validation_path.exists():
                        validation = _parse(validation_path.read_bytes())
                can_retry = isinstance(error, APIError) and error.status in (429, 529)
                result = {"attempt_id": attempt_id, "finished_utc": _utc(),
                          "status": "success" if error is None else "retryable" if can_retry else "terminal",
                          "error_type": type(error).__name__ if error is not None else None,
                          "http_status": capture.get("http_status"),
                          "observation_path": metadata["archive_record"] if metadata else None,
                          "usage": _usage(validation.get("usage")),
                          "body_complete": capture.get("body_complete")}
                if can_retry:
                    server_delay = retry_after_seconds(capture.get("safe_headers", {}).get("retry-after"), clock.time())
                    fallback = min(60, 2 * 2 ** index) + random_value()
                    result["retry_delay_seconds"] = max(server_delay or 0, fallback)
                    limiter.throttle(result["retry_delay_seconds"])
                _write_once(attempts_dir / (attempt_id + ".result.json"), result)
                with mutex:
                    entry["result"] = result
                    in_flight.discard(digest)
                if error is None:
                    complete(digest, response, metadata)
                    limiter.success(estimated, response["usage"]["input_tokens"])
                    return
                if not can_retry or index >= MAX_RETRIES:
                    with mutex:
                        failures[digest] = "Retry limit exhausted" if can_retry else type(error).__name__
                    stop.set()
                    return
                with mutex:
                    retrying.add(digest)
                report()
        finally:
            with mutex:
                in_flight.discard(digest)
                retrying.discard(digest)
            report()

    if failures:
        phase = "failed"
        return report()
    if not run and not (output / "targets.json").exists():
        return report()
    if run:
        pending = [digest for digest in grouped if digest not in responses]
        if pending:
            key = key or load_key()
            phase = "running"
            report()
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jev-collector") as pool:
                futures = {pool.submit(work, digest) for digest in pending}
                try:
                    while futures:
                        done, futures = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                        report()
                except BaseException:
                    stop.set()
                    phase = "failed"
                    report()
                    raise
    if failures or len(responses) != len(grouped):
        phase = "failed"
        return report()
    from experiments.calibrated_targets import load_targets, write_targets

    rows = [row for digest, cases in grouped.items() for row in _rows(cases, responses[digest], observations[digest])]
    teacher = {"model": model, "provider": "TypeSafe", "protocol": PROTOCOL,
               "schema_fingerprint": manifest["schema_fingerprint"]}
    targets_path = output / "targets.json"
    if targets_path.exists():
        existing = load_targets(targets_path, suite_path)
        if existing["teacher"] != teacher:
            raise ValueError("Completed target teacher metadata differs")
        comparable = [{k: row[k] for k in ("case_id", "question_id", "probabilities", "request_sha256", "observation_path")}
                      for row in existing["rows"]]
        order = lambda row: (row["case_id"], row["question_id"])
        if sorted(comparable, key=order) != sorted(rows, key=order):
            raise ValueError("Completed targets differ from validated archive observations")
    else:
        write_targets(suite_path, rows, teacher, targets_path)
    phase = "complete"
    return report()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--archive", type=Path, help="Explicit permitted partition archive; default OUTPUT/archive")
    parser.add_argument("--run", action="store_true", help="Send paid requests; default is offline preparation")
    args = parser.parse_args(argv)
    try:
        result = collect(args.suite, args.output, run=args.run, model=args.model,
                         workers=args.workers, archive=args.archive)
    except Exception as error:
        # Exception strings from network/key configuration are deliberately omitted.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__,
                          "detail": "Collection stopped; inspect preserved output and archives."}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
