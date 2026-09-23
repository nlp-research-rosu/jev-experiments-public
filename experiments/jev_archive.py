"""Durable Jev response capture and conservative reuse of pinned model responses.

Each attempt has immutable request/capture/validation records. The response body
is appended and synced while it arrives, before parsing or scoring. No archive
entry is evicted; the 2 MB bound applies to parsing, never to body preservation.
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "jev-1.13.0"
DEFAULT_DIRECTORY = Path("data/jev-responses")
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_PARSE_BYTES = 2_000_000
SAFE_HEADERS = {"request-id", "x-request-id", "retry-after", "content-type"}
VALIDATION_POLICY_VERSION = 3


class APIError(RuntimeError):
    def __init__(self, status):
        self.status = status
        super().__init__(f"TypeSafe HTTP status {status}; no automatic retry was made.")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _body(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_once(path, value):
    """Publish a complete, fsynced file atomically, refusing replacement."""
    data = value if isinstance(value, bytes) else _body(value) + b"\n"
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            os.chmod(temporary, 0o600)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fixed(model):
    return isinstance(model, str) and re.fullmatch(r"jev-\d+\.\d+\.\d+", model) is not None


def _parse(raw):
    def nonfinite(value):
        raise ValueError("Non-finite JSON value")

    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    return json.loads(raw, parse_constant=nonfinite, object_pairs_hook=unique)


def _validate(payload, response):
    # Reuse the evaluator's primitive checks without consulting authored labels.
    from experiments.jev_replay import answer_view

    if not isinstance(response, dict) or not _fixed(response.get("model")):
        raise ValueError("Missing fixed returned model version")
    if _fixed(payload.get("model")) and payload["model"] != response["model"]:
        raise ValueError("Returned model does not match requested fixed version")
    questions, answers = payload.get("questions"), response.get("answers")
    if not isinstance(questions, dict) or not questions or not isinstance(answers, dict) or set(questions) != set(answers):
        raise ValueError("Missing or unexpected answer IDs")
    normalized = 0
    for qid, question in questions.items():
        if not isinstance(question, dict):
            raise ValueError("Invalid question")
        if question.get("type") == "noul":
            expected = False
        elif question.get("type") == "choice" and isinstance(question.get("criteria"), dict) and question["criteria"]:
            expected = next(iter(question["criteria"]))
        elif question.get("type") == "score":
            expected = 0
        else:
            raise ValueError("Only Noul, Choice and Score responses can be reused")
        normalized += bool(answer_view(question, expected, answers[qid]).get("probability_normalized"))
    usage = response.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in ("input_tokens", "output_tokens")):
        raise ValueError("Invalid token usage")
    return normalized


def _validation(attempt, capture, payload):
    """Capture must already be durable before this function is called."""
    result = {
        "status": "invalid", "validated_utc": _now(), "returned_model": None, "usage": None,
        "validation_policy_version": VALIDATION_POLICY_VERSION,
    }
    try:
        if not capture["body_complete"]:
            raise ValueError("Incomplete response body")
        body_path = attempt / "response.body"
        if body_path.stat().st_size > MAX_PARSE_BYTES:
            raise ValueError("Response exceeds parsing bound; full body is archived")
        parsed = _parse(body_path.read_bytes())
        if isinstance(parsed, dict):
            result["returned_model"] = parsed.get("model")
            result["usage"] = parsed.get("usage")
        if capture["http_status"] != 200:
            raise APIError(capture["http_status"])
        result["normalized_probability_vectors"] = _validate(payload, parsed)
        result["status"] = "valid"
        return parsed
    except BaseException as error:
        result["error_type"] = type(error).__name__
        # HTTP errors remain HTTP errors even when their body is not JSON.
        if capture["http_status"] not in (None, 200):
            result["error_type"] = "APIError"
            raise APIError(capture["http_status"]) from None
        raise
    finally:
        _write_once(attempt / "validation.json", result)


class _RedactedWriter:
    """Redact literal credential bytes, including matches split across chunks."""
    def __init__(self, stream, key):
        self.stream, self.key, self.pending = stream, key.encode(), b""

    def _write(self, chunk):
        if chunk:
            self.stream.write(chunk)
            self.stream.flush()
            os.fsync(self.stream.fileno())

    def append(self, chunk):
        self.pending += chunk
        while len(self.pending) >= len(self.key):
            position = self.pending.find(self.key)
            if position >= 0:
                self._write(self.pending[:position] + b"[REDACTED]")
                self.pending = self.pending[position + len(self.key):]
            else:
                split = len(self.pending) - len(self.key) + 1
                self._write(self.pending[:split])
                self.pending = self.pending[split:]
                break

    def finish(self):
        self._write(self.pending.replace(self.key, b"[REDACTED]"))
        self.pending = b""


class ArchiveClient:
    def __init__(self, key, directory=DEFAULT_DIRECTORY, *, fresh=False):
        self.key, self.directory, self.fresh = key, Path(directory), fresh
        self.last_record = None

    def _metadata(self, attempt, capture, *, cache_hit, returned_model=None):
        self.last_record = {
            "archive_record": str((attempt / "capture.json").resolve()),
            "cache_hit": cache_hit, "requested_model": capture["requested_model"],
            "returned_model": returned_model, "request_sha256": capture["request_sha256"],
            "http_status": capture["http_status"], "elapsed_ms": capture["elapsed_ms"],
        }

    def _cached(self, payload):
        if self.fresh or not _fixed(payload.get("model")):
            return None
        candidates = []
        for path in self.directory.glob("*/capture.json"):
            try:
                capture = _parse(path.read_bytes())
                if not isinstance(capture, dict) or not isinstance(capture["started_utc"], str):
                    continue
                provenance = capture.get("provenance", {})
                if not isinstance(provenance, dict):
                    continue
                response_line = provenance.get("response_line", 0)
                if type(response_line) is not int or response_line < 0:
                    continue
                order = (capture["started_utc"], response_line, str(path))
                candidates.append((order, path, capture))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        for _, path, capture in sorted(candidates, key=lambda item: item[0]):
            try:
                attempt = path.parent
                validation = _parse((attempt / "validation.json").read_bytes())
                if capture["format_version"] != 1 or capture["endpoint"] != ENDPOINT or capture["source"] not in ("live", "import"):
                    continue
                if validation["status"] not in ("valid", "invalid") or capture["http_status"] != 200 or not capture["body_complete"]:
                    continue
                if capture.get("request_redacted"):
                    continue
                request_body = capture["request_body"].encode()
                if _hash(request_body) != capture["request_sha256"]:
                    continue
                if capture["source"] == "live" and capture["wire_request_sha256"] != capture["request_sha256"]:
                    continue
                original = _parse(request_body)
                if original.get("model") != capture["requested_model"]:
                    continue
                # Alias captures can serve a pin only when all other inputs match.
                if _body({**original, "model": payload["model"]}) != _body(payload):
                    continue
                body_path = attempt / "response.body"
                if body_path.stat().st_size > MAX_PARSE_BYTES or _file_hash(body_path) != capture["response_sha256"]:
                    continue
                parsed = _parse(body_path.read_bytes())
                if capture["source"] == "import":
                    provenance = capture["provenance"]
                    request_line = (attempt / "source-request.jsonl").read_bytes()
                    response_line = (attempt / "source-response.jsonl").read_bytes()
                    identity = _hash(_body({
                        "request_path": provenance["request_path"],
                        "response_path": provenance["response_path"],
                        "line": provenance["response_line"],
                    }) + request_line + response_line)
                    saved_request, saved_response = _parse(request_line), _parse(response_line)
                    if attempt.name != "import-" + identity or provenance["raw_http_body_available"] is not False:
                        continue
                    if saved_request["case_id"] != saved_response["case_id"] or saved_request["case_id"] != provenance["case_id"]:
                        continue
                    if _body(saved_request["request"]) != request_body or _hash(_body(saved_response["response"])) != capture["response_sha256"]:
                        continue
                _validate(original, parsed)
                normalized = _validate(payload, parsed)
                if validation["returned_model"] != parsed["model"]:
                    continue
                if validation["status"] == "invalid":
                    sidecar = attempt / f"revalidation-v{VALIDATION_POLICY_VERSION}.json"
                    if not sidecar.exists():
                        _write_once(sidecar, {
                            "status": "valid", "original_validation_status": "invalid",
                            "validation_policy_version": VALIDATION_POLICY_VERSION,
                            "original_validation_sha256": _file_hash(attempt / "validation.json"),
                            "capture_sha256": _file_hash(path), "response_sha256": capture["response_sha256"],
                            "normalized_probability_vectors": normalized, "validated_utc": _now(),
                            "returned_model": parsed["model"], "usage": parsed["usage"],
                        })
                self._metadata(attempt, capture, cache_hit=True, returned_model=parsed["model"])
                self.last_record.update(
                    original_validation_status=validation["status"], current_validation_status="valid",
                    validation_policy_version=VALIDATION_POLICY_VERSION, revalidated=True,
                )
                return parsed
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                continue
        return None

    def __call__(self, payload):
        self.last_record = None
        request_body = _body(payload)
        cached = self._cached(payload)
        if cached is not None:
            return cached
        if not self.key or not isinstance(self.key, str) or not self.key.isascii() or any(c.isspace() for c in self.key):
            raise ValueError("A valid API key is required for an uncached request")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        attempt = self.directory / (_now().replace(":", "-") + "-" + uuid.uuid4().hex)
        attempt.mkdir(mode=0o700)
        _sync_directory(self.directory)
        safe_body = request_body.replace(self.key.encode(), b"[REDACTED]")
        capture = {
            "format_version": 1, "source": "live", "endpoint": ENDPOINT,
            "attempt_id": attempt.name, "started_utc": _now(),
            "requested_model": payload.get("model"), "request_body": safe_body.decode(),
            "request_sha256": _hash(safe_body), "wire_request_sha256": _hash(request_body),
            "request_redacted": safe_body != request_body,
            "http_status": None, "safe_headers": {}, "body_complete": False,
        }
        _write_once(attempt / "request.json", capture)
        started = time.perf_counter()
        connection = http.client.HTTPSConnection("api.typesafe.ai", timeout=45)
        transport_error = None
        with (attempt / "response.body").open("xb") as stream:
            os.chmod(attempt / "response.body", 0o600)
            writer = _RedactedWriter(stream, self.key)
            try:
                connection.request("POST", "/v1/systemone", body=request_body, headers={
                    "Authorization": "Bearer " + self.key, "Content-Type": "application/json",
                })
                remote = connection.getresponse()
                capture["http_status"] = remote.status
                capture["safe_headers"] = {
                    name.lower(): value.replace(self.key, "[REDACTED]")
                    for name, value in remote.getheaders() if name.lower() in SAFE_HEADERS
                }
                _write_once(attempt / "headers.json", {
                    "http_status": capture["http_status"], "safe_headers": capture["safe_headers"],
                })
                while True:
                    chunk = remote.read1(65536)
                    if not chunk:
                        if getattr(remote, "length", None):
                            raise http.client.IncompleteRead(b"", remote.length)
                        break
                    writer.append(chunk)
                capture["body_complete"] = True
            except BaseException as error:
                if isinstance(error, http.client.IncompleteRead):
                    writer.append(error.partial)
                transport_error = error
                capture["transport_error"] = type(error).__name__
            finally:
                writer.finish()
                connection.close()
        capture.update(
            received_utc=_now(), elapsed_ms=(time.perf_counter() - started) * 1000,
            response_sha256=_file_hash(attempt / "response.body"),
            response_bytes=(attempt / "response.body").stat().st_size,
        )
        _write_once(attempt / "capture.json", capture)
        self._metadata(attempt, capture, cache_hit=False)
        if transport_error is not None:
            _write_once(attempt / "validation.json", {
                "status": "invalid", "error_type": type(transport_error).__name__, "validated_utc": _now(),
                "returned_model": None, "usage": None,
                "validation_policy_version": VALIDATION_POLICY_VERSION,
            })
            raise transport_error
        parsed = _validation(attempt, capture, payload)
        self.last_record.update(
            returned_model=parsed["model"], original_validation_status="valid", current_validation_status="valid",
            validation_policy_version=VALIDATION_POLICY_VERSION, revalidated=False,
        )
        return parsed


def import_run(source, directory=DEFAULT_DIRECTORY):
    """Import old parsed JSONL artifacts without pretending they contain HTTP bytes."""
    source, directory = Path(source).resolve(), Path(directory)
    request_path, response_path = source / "requests.jsonl", source / "responses.jsonl"
    requests = {}
    for number, line in enumerate(request_path.read_bytes().splitlines(keepends=True), 1):
        row = _parse(line)
        if row["case_id"] in requests:
            raise ValueError("Ambiguous duplicate request case ID in import")
        requests[row["case_id"]] = (row["request"], line, number)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    counts = {"imported": 0, "existing": 0}
    for number, line in enumerate(response_path.read_bytes().splitlines(keepends=True), 1):
        row = _parse(line)
        payload, request_line, request_number = requests[row["case_id"]]
        identity = _hash(_body({"request_path": str(request_path), "response_path": str(response_path), "line": number}) + request_line + line)
        attempt = directory / ("import-" + identity)
        if attempt.exists():
            if (attempt / "source-request.jsonl").read_bytes() != request_line or (attempt / "source-response.jsonl").read_bytes() != line:
                raise ValueError("Existing import provenance differs")
            counts["existing"] += 1
            continue
        staged = directory / (".import-" + identity + "-" + uuid.uuid4().hex)
        staged.mkdir(mode=0o700)
        request_body, response_body = _body(payload), _body(row["response"])
        capture = {
            "format_version": 1, "source": "import", "endpoint": ENDPOINT,
            "attempt_id": attempt.name, "started_utc": row.get("received_utc", _now()),
            "received_utc": row.get("received_utc"), "elapsed_ms": row.get("elapsed_ms"),
            "requested_model": payload.get("model"), "request_body": request_body.decode(),
            "request_sha256": _hash(request_body), "wire_request_sha256": None,
            "request_redacted": False, "http_status": 200, "safe_headers": {},
            "body_complete": True, "response_sha256": _hash(response_body), "response_bytes": len(response_body),
            "provenance": {
                "request_path": str(request_path), "request_line": request_number,
                "response_path": str(response_path), "response_line": number,
                "case_id": row["case_id"], "imported_utc": _now(),
                "raw_http_body_available": False,
                "http_status_source": "inferred from legacy runner saving only HTTP 200 responses",
                "body_source": "reconstructed JSON; exact original artifact lines saved separately",
            },
        }
        _write_once(staged / "source-request.jsonl", request_line)
        _write_once(staged / "source-response.jsonl", line)
        _write_once(staged / "response.body", response_body)
        _write_once(staged / "capture.json", capture)
        try:
            _validation(staged, capture, payload)
        except (ValueError, APIError):
            pass  # Preserve invalid historical responses, without making them cacheable.
        staged.rename(attempt)
        _sync_directory(directory)
        counts["imported"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()
    print(json.dumps(import_run(args.import_run, args.archive)))


if __name__ == "__main__":
    main()
