"""Archive record serialisation and hashing."""

import pytest
from pydantic import ValidationError

from open_context.archive import ArchiveRecord, canonical_json, hash_body


def test_hash_is_computed_on_construction():
    record = ArchiveRecord(seq=0, id="msg_1", payload={"a": 1})
    assert record.hash == hash_body(0, "msg_1", "message", {"a": 1})
    assert record.verify()


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_identical_content_hashes_identically():
    first = ArchiveRecord(seq=3, id="msg_1", payload={"x": [1, 2], "y": "z"})
    second = ArchiveRecord(seq=3, id="msg_1", payload={"y": "z", "x": [1, 2]})
    assert first.hash == second.hash


def test_hash_covers_every_field():
    base = ArchiveRecord(seq=0, id="msg_1", kind="message", payload={"a": 1})
    assert ArchiveRecord(seq=1, id="msg_1", kind="message", payload={"a": 1}).hash != base.hash
    assert ArchiveRecord(seq=0, id="msg_2", kind="message", payload={"a": 1}).hash != base.hash
    assert ArchiveRecord(seq=0, id="msg_1", kind="tool", payload={"a": 1}).hash != base.hash
    assert ArchiveRecord(seq=0, id="msg_1", kind="message", payload={"a": 2}).hash != base.hash


def test_hashing_is_byte_exact_and_does_not_normalise_unicode():
    """Unlike Evidence, which normalises to NFC so duplicate content deduplicates,
    the archive hash must notice every byte difference. It stores originals, and
    one composed form replacing another is a real change in what was written."""
    assert hash_body(0, "m", "message", "cafe\u0301") != hash_body(0, "m", "message", "caf\u00e9")


def test_line_round_trip():
    record = ArchiveRecord(seq=7, id="msg_7", kind="tool_result", payload={"rows": [1, 2, 3]})
    assert ArchiveRecord.from_line(record.to_line().rstrip(b"\n")) == record


def test_line_is_newline_terminated_and_single_line():
    line = ArchiveRecord(seq=0, id="m", payload={"text": "one\ntwo"}).to_line()
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1, "embedded newlines must be escaped, not written raw"


def test_tampered_line_fails_verification():
    record = ArchiveRecord(seq=0, id="msg_1", payload={"content": "original"})
    tampered = record.to_line().replace(b"original", b"modified!")
    assert not ArchiveRecord.from_line(tampered.rstrip(b"\n")).verify()


def test_payload_may_be_any_json_shape():
    """The archive stores raw provider material whose shape nothing here knows."""
    for payload in [None, 1, "text", [1, {"a": 2}], {"deep": {"nested": [True, None]}}]:
        assert ArchiveRecord(seq=0, id="m", payload=payload).verify()


def test_malformed_lines_are_rejected():
    with pytest.raises(ValueError):
        ArchiveRecord.from_line(b"not json")
    with pytest.raises(ValueError, match="not a JSON object"):
        ArchiveRecord.from_line(b"[1, 2]")
    with pytest.raises(ValueError, match="missing"):
        ArchiveRecord.from_line(b'{"seq": 0}')


def test_records_are_frozen():
    record = ArchiveRecord(seq=0, id="m", payload={})
    with pytest.raises(ValidationError):
        record.seq = 1


def test_negative_seq_rejected():
    with pytest.raises(ValidationError):
        ArchiveRecord(seq=-1, id="m", payload={})
