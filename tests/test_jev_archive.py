import hashlib
import http.client
import json
from pathlib import Path

import pytest


def payload(model="jev-1.13.0"):
    return {
        "model": model, "state": {"result": "done"},
        "questions": {"confirmed": {"type": "noul", "instructions": "Completed?"}},
    }


def response(p=0.8, model="jev-1.13.0"):
    return {
        "model": model, "answers": {"confirmed": {"type": "noul", "noul": p}},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def transport(monkeypatch, bodies, *, status=200, headers=None, interrupt=None, remaining=None):
    """Replace only HTTPS; all persistence, parsing and cache validation stay real."""
    observed = []

    class Connection:
        def __init__(self, host, timeout):
            assert host == "api.typesafe.ai" and timeout > 0

        def request(self, method, path, body, headers):
            observed.append((method, path, body, headers))

        def getresponse(self):
            chunks = iter(bodies.pop(0))

            class Response:
                def getheaders(self):
                    return headers or []

                def read1(self, size):
                    assert size > 0
                    chunk = next(chunks, None)
                    if chunk is None:
                        if interrupt is not None:
                            raise interrupt
                        return b""
                    return chunk

            result = Response()
            result.status = status
            result.length = remaining
            return result

        def close(self):
            pass

    monkeypatch.setattr(http.client, "HTTPSConnection", Connection)
    return observed


def records(directory):
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*/capture.json"))]


def test_durable_capture_then_validated_cache_avoids_second_network_request(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    raw = json.dumps(response(), indent=1).encode()
    sent = transport(monkeypatch, [[raw]], headers=[
        ("X-Request-ID", "request-123"), ("Retry-After", "3"),
        ("Authorization", "Bearer test-secret"), ("Set-Cookie", "sensitive"),
    ])
    archive = tmp_path / "archive"
    client = ArchiveClient("test-secret", archive)
    assert client(payload()) == response()
    first = client.last_record.copy()
    assert first["cache_hit"] is False
    entry = records(archive)[0]
    attempt = Path(first["archive_record"]).parent
    assert (attempt / "response.body").read_bytes() == raw
    assert entry["request_body"] == sent[0][2].decode()
    assert entry["request_sha256"] == hashlib.sha256(sent[0][2]).hexdigest()
    assert entry["http_status"] == 200 and entry["body_complete"] is True
    assert entry["safe_headers"] == {"x-request-id": "request-123", "retry-after": "3"}
    assert entry["requested_model"] == "jev-1.13.0"
    assert entry["started_utc"] and entry["elapsed_ms"] >= 0
    validation = json.loads((attempt / "validation.json").read_text())
    assert validation["returned_model"] == "jev-1.13.0"
    assert validation["usage"] == {"input_tokens": 10, "output_tokens": 2}
    assert client(payload()) == response()
    assert len(sent) == 1 and len(records(archive)) == 1
    assert client.last_record["cache_hit"] is True
    assert client.last_record["archive_record"] == first["archive_record"]
    assert all(b"test-secret" not in p.read_bytes() for p in archive.rglob("*") if p.is_file())


def test_fresh_samples_never_overwrite_and_cache_returns_earliest_valid_draw(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    transport(monkeypatch, [[json.dumps(response(p)).encode()] for p in (0.8, 0.6)])
    fresh = ArchiveClient("test-secret", tmp_path, fresh=True)
    assert fresh(payload()) == response(0.8)
    original = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert fresh(payload()) == response(0.6)
    assert len(records(tmp_path)) == 2
    assert all(p.read_bytes() == data for p, data in original.items())
    cached = ArchiveClient(None, tmp_path)
    assert cached(payload()) == response(0.8)


def test_alias_capture_can_satisfy_exact_pinned_request_but_alias_is_never_reused(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    sent = transport(monkeypatch, [[json.dumps(response(p)).encode()] for p in (0.8, 0.6)])
    client = ArchiveClient("test-secret", tmp_path)
    assert client(payload("jev-latest")) == response(0.8)
    assert client(payload()) == response(0.8)
    assert client.last_record["requested_model"] == "jev-latest"
    assert client.last_record["cache_hit"] is True
    assert client(payload("jev-latest")) == response(0.6)
    assert len(sent) == 2
    changed = payload()
    changed["state"]["result"] = "pending"
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(changed)
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(payload("jev-1.14.0"))


@pytest.mark.parametrize("status,raw", [
    (429, b'{"error":"rate limited","echo":"test-secret"}'),
    (200, b'{"malformed": "test-secret"'),
    (200, b'\xff\xfeinvalid utf8 test-secret'),
    (200, json.dumps(response(float("nan"))).encode()),
    (200, json.dumps(response(model="jev-1.14.0")).encode()),
])
def test_failed_paid_bodies_are_retained_redacted_and_never_reused(tmp_path, monkeypatch, status, raw):
    from experiments.jev_archive import ArchiveClient

    # Splitting the echoed key also checks streaming credential redaction.
    transport(monkeypatch, [[raw[:20], raw[20:35], raw[35:]]], status=status)
    client = ArchiveClient("test-secret", tmp_path)
    with pytest.raises((ValueError, RuntimeError)):
        client(payload())
    entry = records(tmp_path)[0]
    attempt = Path(client.last_record["archive_record"]).parent
    assert (attempt / "response.body").read_bytes() == raw.replace(b"test-secret", b"[REDACTED]")
    assert entry["http_status"] == status
    assert json.loads((attempt / "validation.json").read_text())["status"] == "invalid"
    assert all(b"test-secret" not in p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(payload())


def test_interrupted_stream_preserves_received_body_and_marks_incomplete(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    transport(monkeypatch, [[b'{"partial":"paid'] ], interrupt=KeyboardInterrupt())
    client = ArchiveClient("test-secret", tmp_path)
    with pytest.raises(KeyboardInterrupt):
        client(payload())
    entry = records(tmp_path)[0]
    assert entry["body_complete"] is False and entry["transport_error"] == "KeyboardInterrupt"
    assert (Path(client.last_record["archive_record"]).parent / "response.body").read_bytes() == b'{"partial":"paid'
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(payload())


def test_truncated_http_stream_retains_incomplete_read_partial_bytes(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    transport(monkeypatch, [[b"first "]], interrupt=http.client.IncompleteRead(b"last", 10))
    client = ArchiveClient("test-secret", tmp_path)
    with pytest.raises(http.client.IncompleteRead):
        client(payload())
    assert (Path(client.last_record["archive_record"]).parent / "response.body").read_bytes() == b"first last"
    assert records(tmp_path)[0]["body_complete"] is False


def test_capture_exists_before_semantic_validation_and_invalid_schema_is_not_cached(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    invalid = response()
    invalid["answers"]["confirmed"]["noul"] = 2.0
    transport(monkeypatch, [[json.dumps(invalid).encode()]])
    client = ArchiveClient("test-secret", tmp_path)
    with pytest.raises(ValueError):
        client(payload())
    assert len(records(tmp_path)) == 1
    assert json.loads((Path(client.last_record["archive_record"]).parent / "response.body").read_text()) == invalid
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(payload())


def test_cache_revalidates_capture_instead_of_trusting_success_marker(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    transport(monkeypatch, [[json.dumps(response()).encode()]])
    client = ArchiveClient("test-secret", tmp_path)
    client(payload())
    attempt = Path(client.last_record["archive_record"]).parent
    (attempt / "response.body").write_text(json.dumps(response(0.1)))
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(payload())


def test_import_keeps_original_lines_and_alias_provenance_idempotently(tmp_path):
    from experiments.jev_archive import ArchiveClient, import_run

    source, archive = tmp_path / "old-run", tmp_path / "archive"
    source.mkdir()
    request_line = json.dumps({"case_id": "case", "request": payload("jev-latest")}) + "\n"
    response_line = json.dumps({
        "case_id": "case", "response": response(), "elapsed_ms": 123.0,
        "received_utc": "2026-09-20T21:08:38+00:00",
    }) + "\n"
    (source / "requests.jsonl").write_text(request_line)
    (source / "responses.jsonl").write_text(response_line)
    assert import_run(source, archive) == {"imported": 1, "existing": 0}
    original = {p: p.read_bytes() for p in archive.rglob("*") if p.is_file()}
    assert import_run(source, archive) == {"imported": 0, "existing": 1}
    assert all(p.read_bytes() == data for p, data in original.items())
    client = ArchiveClient(None, archive)
    assert client(payload()) == response()
    entry = records(archive)[0]
    attempt = Path(client.last_record["archive_record"]).parent
    assert entry["requested_model"] == "jev-latest"
    assert entry["provenance"]["request_path"] == str((source / "requests.jsonl").resolve())
    assert entry["provenance"]["raw_http_body_available"] is False
    assert (attempt / "source-request.jsonl").read_text() == request_line
    assert (attempt / "source-response.jsonl").read_text() == response_line


def test_import_preserves_repeated_draws_for_same_case(tmp_path):
    from experiments.jev_archive import ArchiveClient, import_run

    source, archive = tmp_path / "run", tmp_path / "archive"
    source.mkdir()
    (source / "requests.jsonl").write_text(json.dumps({"case_id": "case", "request": payload()}) + "\n")
    (source / "responses.jsonl").write_text("".join(json.dumps({
        "case_id": "case", "response": response(p), "elapsed_ms": 1, "received_utc": "earlier",
    }) + "\n" for p in (0.8, 0.6)))
    assert import_run(source, archive) == {"imported": 2, "existing": 0}
    assert len(records(archive)) == 2
    assert import_run(source, archive) == {"imported": 0, "existing": 2}
    assert ArchiveClient(None, archive)(payload()) == response(0.8)


def test_content_length_truncation_is_not_mistaken_for_complete_json(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    raw = json.dumps(response()).encode()
    transport(monkeypatch, [[raw]], remaining=17)
    client = ArchiveClient("test-secret", tmp_path)
    with pytest.raises(http.client.IncompleteRead):
        client(payload())
    assert records(tmp_path)[0]["body_complete"] is False
    assert (Path(client.last_record["archive_record"]).parent / "response.body").read_bytes() == raw


def test_redacted_request_never_matches_an_actual_redaction_marker_request(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    transport(monkeypatch, [[json.dumps(response()).encode()]])
    request = payload()
    request["state"] = {"value": "test-secret"}
    ArchiveClient("test-secret", tmp_path)(request)
    request["state"] = {"value": "[REDACTED]"}
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(request)


def test_full_oversized_body_is_saved_despite_parse_bound(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    raw = b" " * 2_000_010 + json.dumps(response()).encode()
    transport(monkeypatch, [[raw]])
    client = ArchiveClient("test-secret", tmp_path)
    with pytest.raises(ValueError, match="parsing bound"):
        client(payload())
    assert (Path(client.last_record["archive_record"]).parent / "response.body").read_bytes() == raw


def test_captured_body_is_durable_before_validator_is_entered(tmp_path, monkeypatch):
    from experiments import jev_archive

    real_validate = jev_archive._validate
    transport(monkeypatch, [[json.dumps(response()).encode()]])

    def inspect_then_validate(request, result):
        captures = list(tmp_path.glob("*/capture.json"))
        assert len(captures) == 1
        assert json.loads((captures[0].parent / "response.body").read_text()) == response()
        return real_validate(request, result)

    monkeypatch.setattr(jev_archive, "_validate", inspect_then_validate)
    assert jev_archive.ArchiveClient("test-secret", tmp_path)(payload()) == response()


@pytest.mark.parametrize("field,value", [
    ("endpoint", "https://different.example/v1/systemone"),
    ("format_version", 999),
    ("source", "unknown"),
    ("wire_request_sha256", "bad-digest"),
])
def test_cache_requires_known_endpoint_format_and_request_provenance(tmp_path, monkeypatch, field, value):
    from experiments.jev_archive import ArchiveClient

    transport(monkeypatch, [[json.dumps(response()).encode()]])
    client = ArchiveClient("test-secret", tmp_path)
    client(payload())
    record_path = Path(client.last_record["archive_record"])
    record = json.loads(record_path.read_text())
    record[field] = value
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(payload())


def test_imported_cache_checks_preserved_source_provenance(tmp_path):
    from experiments.jev_archive import ArchiveClient, import_run

    source, archive = tmp_path / "run", tmp_path / "archive"
    source.mkdir()
    (source / "requests.jsonl").write_text(json.dumps({"case_id": "case", "request": payload("jev-latest")}) + "\n")
    (source / "responses.jsonl").write_text(json.dumps({
        "case_id": "case", "response": response(), "elapsed_ms": 1, "received_utc": "earlier",
    }) + "\n")
    import_run(source, archive)
    next(archive.glob("*/source-response.jsonl")).write_text(json.dumps({"case_id": "case", "response": response(0.1)}))
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, archive)(payload())


@pytest.mark.parametrize("malformed_provenance", [
    None, [], {"response_line": "2"}, {"response_line": {}},
    {"response_line": None}, {"response_line": False},
    {"response_line": -0.5}, {"response_line": -1},
])
def test_bad_sort_metadata_is_preserved_and_skipped_before_earliest_valid_reuse(
    tmp_path, monkeypatch, malformed_provenance,
):
    from experiments.jev_archive import ArchiveClient

    transport(monkeypatch, [[json.dumps(response(p)).encode()] for p in (0.1, 0.8, 0.6)])
    client = ArchiveClient("test-secret", tmp_path, fresh=True)
    captures = []
    for _ in range(3):
        client(payload())
        captures.append(Path(client.last_record["archive_record"]))
    # Equal timestamps force tie-breaking metadata to be compared.
    for path in captures:
        capture = json.loads(path.read_text())
        capture["started_utc"] = "2026-09-20T21:08:38+00:00"
        if path == captures[0]:
            capture["provenance"] = malformed_provenance
        path.write_text(json.dumps(capture))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    offline = ArchiveClient(None, tmp_path)
    assert offline(payload()) == response(0.8)
    assert offline.last_record["archive_record"] == str(captures[1])
    assert before == {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}


def test_original_invalid_cent_rounded_response_is_revalidated_without_new_attempt(tmp_path, monkeypatch):
    from experiments.jev_archive import ArchiveClient

    # Hand-authored old-format capture of the observed 0.99 Choice vector.
    request = payload()
    request["questions"] = {"status": {"type": "choice", "criteria": {
        "unknown": "Unknown", "no_effect": "No effect", "reversed": "Reversed", "effective": "Effective",
    }}}
    parsed = {"model": "jev-1.13.0", "answers": {"status": {
        "type": "choice", "choice": "effective", "confidence": 0.91,
        "probabilities": {"unknown": 0.05, "no_effect": 0.0, "reversed": 0.01, "effective": 0.93},
    }}, "usage": {"input_tokens": 1443, "output_tokens": 148}}
    request_body, response_body = json.dumps(request).encode(), json.dumps(parsed).encode()
    attempt = tmp_path / "original-paid-attempt"
    attempt.mkdir()
    (attempt / "response.body").write_bytes(response_body)
    (attempt / "capture.json").write_text(json.dumps({
        "format_version": 1, "source": "live", "endpoint": "https://api.typesafe.ai/v1/systemone",
        "started_utc": "2026-09-20T21:27:10.275900+00:00", "elapsed_ms": 370,
        "http_status": 200, "body_complete": True, "requested_model": "jev-1.13.0",
        "request_body": request_body.decode(), "request_sha256": hashlib.sha256(request_body).hexdigest(),
        "wire_request_sha256": hashlib.sha256(request_body).hexdigest(),
        "response_sha256": hashlib.sha256(response_body).hexdigest(),
    }))
    (attempt / "validation.json").write_text(json.dumps({
        "status": "invalid", "error_type": "ValueError", "returned_model": "jev-1.13.0", "usage": parsed["usage"],
    }))
    before = {path: path.read_bytes() for path in attempt.iterdir()}

    def forbidden_https(*args, **kwargs):
        pytest.fail("Revalidating an archived paid response must never make a network request")

    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden_https)
    client = ArchiveClient(None, tmp_path)
    assert client(request) == parsed
    assert client.last_record["cache_hit"] is True
    assert client.last_record["original_validation_status"] == "invalid"
    assert client.last_record["current_validation_status"] == "valid"
    assert client.last_record["validation_policy_version"] == 3
    assert client.last_record["revalidated"] is True
    assert len(records(tmp_path)) == 1
    assert all(path.read_bytes() == raw for path, raw in before.items())
    sidecar = attempt / "revalidation-v3.json"
    verdict = json.loads(sidecar.read_text())
    assert verdict["original_validation_status"] == "invalid" and verdict["status"] == "valid"
    assert verdict["validation_policy_version"] == 3 and verdict["normalized_probability_vectors"] == 1
    sidecar_before = sidecar.read_bytes()
    assert client(request) == parsed
    assert sidecar.read_bytes() == sidecar_before


def test_score_archive_reuses_earliest_exact_version_and_keeps_all_samples(tmp_path, monkeypatch):
    from test_jev_replay import score_fixture

    from experiments.jev_archive import ArchiveClient

    question, answer = score_fixture()
    request = {"model": "jev-latest", "state": "verified complete", "questions": {"rating": question}}
    first = {"model": "jev-1.13.0", "answers": {"rating": answer}, "usage": {"input_tokens": 5, "output_tokens": 2}}
    second = json.loads(json.dumps(first))
    second["answers"]["rating"].update(score=1.5, probabilities={"0": 0.2, "1": 0.1, "2": 0.7})
    raw_bodies = [json.dumps(item, indent=2).encode() for item in (first, second)]
    sent = transport(monkeypatch, [[raw] for raw in raw_bodies])
    client = ArchiveClient("test-secret", tmp_path, fresh=True)
    assert client(request) == first
    assert client(request) == second
    preserved = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    cached = ArchiveClient(None, tmp_path)
    assert cached({**request, "model": "jev-1.13.0"}) == first
    assert cached.last_record["cache_hit"] is True
    assert len(sent) == 2 and len(records(tmp_path)) == 2
    assert all(path.read_bytes() == raw for path, raw in preserved.items())
    assert sorted(path.read_bytes() for path in tmp_path.glob("*/response.body")) == sorted(raw_bodies)
    with pytest.raises(ValueError, match="key"):
        cached({**request, "model": "jev-1.14.0"})


def test_score_old_policy_revalidation_preserves_prior_verdicts_and_checks_body_hash(tmp_path, monkeypatch):
    from test_jev_replay import score_fixture

    from experiments.jev_archive import ArchiveClient

    question, answer = score_fixture()
    request = {"model": "jev-1.13.0", "state": "verified complete", "questions": {"rating": question}}
    parsed = {"model": "jev-1.13.0", "answers": {"rating": answer}, "usage": {"input_tokens": 5, "output_tokens": 2}}
    request_body, response_body = json.dumps(request).encode(), json.dumps(parsed).encode()
    attempt = tmp_path / "paid-score-attempt"
    attempt.mkdir()
    (attempt / "response.body").write_bytes(response_body)
    (attempt / "capture.json").write_text(json.dumps({
        "format_version": 1, "source": "live", "endpoint": "https://api.typesafe.ai/v1/systemone",
        "started_utc": "2026-09-20T21:27:10.275900+00:00", "elapsed_ms": 370,
        "http_status": 200, "body_complete": True, "requested_model": "jev-1.13.0",
        "request_body": request_body.decode(), "request_sha256": hashlib.sha256(request_body).hexdigest(),
        "wire_request_sha256": hashlib.sha256(request_body).hexdigest(),
        "response_sha256": hashlib.sha256(response_body).hexdigest(),
    }))
    (attempt / "validation.json").write_text(json.dumps({
        "status": "invalid", "validation_policy_version": 1, "error_type": "ValueError",
        "returned_model": "jev-1.13.0", "usage": parsed["usage"],
    }))
    (attempt / "revalidation-v2.json").write_text(json.dumps({
        "status": "invalid", "validation_policy_version": 2, "error_type": "ValueError",
    }))
    before = {path: path.read_bytes() for path in attempt.iterdir()}

    def forbidden_https(*args, **kwargs):
        pytest.fail("Offline Score revalidation must never make a network request")

    monkeypatch.setattr(http.client, "HTTPSConnection", forbidden_https)
    cached = ArchiveClient(None, tmp_path)
    assert cached(request) == parsed
    assert all(path.read_bytes() == raw for path, raw in before.items())
    verdict = json.loads((attempt / "revalidation-v3.json").read_text())
    assert verdict["status"] == "valid" and verdict["original_validation_status"] == "invalid"
    assert verdict["original_validation_sha256"] == hashlib.sha256(before[attempt / "validation.json"]).hexdigest()
    assert verdict["capture_sha256"] == hashlib.sha256(before[attempt / "capture.json"]).hexdigest()
    assert verdict["response_sha256"] == hashlib.sha256(response_body).hexdigest()
    assert cached.last_record["validation_policy_version"] == 3
    # A successful current sidecar cannot bypass verification of the stored body.
    (attempt / "response.body").write_bytes(response_body + b" ")
    with pytest.raises(ValueError, match="key"):
        cached(request)


def test_invalid_score_is_retained_before_rejection_and_never_reused(tmp_path, monkeypatch):
    from test_jev_replay import score_fixture

    from experiments.jev_archive import ArchiveClient

    question, answer = score_fixture()
    answer["probabilities"] = {"0": 0.2, "2": 0.8}
    request = {"model": "jev-1.13.0", "state": "verified complete", "questions": {"rating": question}}
    raw = json.dumps({"model": "jev-1.13.0", "answers": {"rating": answer}, "usage": {"input_tokens": 5, "output_tokens": 2}}).encode()
    transport(monkeypatch, [[raw]])
    client = ArchiveClient("test-secret", tmp_path)
    with pytest.raises(ValueError):
        client(request)
    attempt = Path(client.last_record["archive_record"]).parent
    assert (attempt / "response.body").read_bytes() == raw
    assert json.loads((attempt / "validation.json").read_text())["status"] == "invalid"
    with pytest.raises(ValueError, match="key"):
        ArchiveClient(None, tmp_path)(request)
