"""Evidence and content hashing."""

import pytest
from pydantic import ValidationError

from open_context.models import Evidence, Session, SourceType, content_hash, ids


def make_evidence(**overrides):
    defaults = {
        "session_id": Session().id,
        "source_ids": [ids.new_id(ids.MESSAGE)],
        "source_type": SourceType.MESSAGE,
        "content": "We are using RocksDB for the local store.",
    }
    return Evidence(**{**defaults, **overrides})


def test_hash_computed_on_construction():
    evidence = make_evidence()
    assert evidence.content_hash == content_hash(evidence.content)
    assert evidence.verify()


def test_identical_content_hashes_identically():
    assert make_evidence().content_hash == make_evidence().content_hash


def test_unicode_normalisation_makes_equivalent_text_hash_alike():
    """Composed and decomposed forms of the same text must deduplicate together."""
    assert content_hash("cafe\u0301") == content_hash("caf\u00e9")


def test_mismatched_hash_rejected():
    with pytest.raises(ValidationError, match="does not match content"):
        make_evidence(content_hash="0" * 64)


def test_matching_hash_accepted():
    text = "explicit hash"
    evidence = make_evidence(content=text, content_hash=content_hash(text))
    assert evidence.verify()


def test_round_trip_preserves_hash():
    original = make_evidence()
    restored = Evidence.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.verify()


def test_evidence_requires_at_least_one_source():
    with pytest.raises(ValidationError):
        make_evidence(source_ids=[])


def test_evidence_rejects_duplicate_sources():
    ref = ids.new_id(ids.MESSAGE)
    with pytest.raises(ValidationError, match="must not contain duplicates"):
        make_evidence(source_ids=[ref, ref])


def test_evidence_rejects_snapshot_as_source():
    with pytest.raises(ValidationError, match="expected one of"):
        make_evidence(source_ids=[ids.new_id(ids.SNAPSHOT)])


def test_empty_content_rejected():
    """Evidence exists to hold exact source text. Empty evidence proves nothing."""
    with pytest.raises(ValidationError):
        make_evidence(content="")
