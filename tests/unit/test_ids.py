"""Identifier scheme."""

import pytest

from open_context.models import ids


def test_new_id_has_prefix_and_is_valid():
    value = ids.new_id(ids.MESSAGE)
    assert value.startswith("msg_")
    assert ids.is_valid(value)
    assert ids.prefix_of(value) == "msg"


def test_ids_are_unique():
    generated = {ids.new_id(ids.MESSAGE) for _ in range(2000)}
    assert len(generated) == 2000


def test_unknown_prefix_rejected():
    with pytest.raises(ValueError, match="unknown id prefix"):
        ids.new_id("nope")


@pytest.mark.parametrize("bad", ["", "msg", "msg_", "msg_XYZ", "MSG_" + "a" * 24, "_" + "a" * 24])
def test_malformed_ids_rejected(bad):
    assert not ids.is_valid(bad)
    with pytest.raises(ValueError):
        ids.prefix_of(bad)


def test_unknown_prefix_is_well_formed_but_invalid():
    assert not ids.is_valid("zzz_" + "a" * 24)


def test_validate_id_enforces_expected_prefix():
    message = ids.new_id(ids.MESSAGE)
    assert ids.validate_id(message, expected=ids.MESSAGE) == message
    with pytest.raises(ValueError, match="expected an id with prefix"):
        ids.validate_id(message, expected=ids.EVIDENCE)


def test_validate_ref_restricts_to_allowed_prefixes():
    evidence = ids.new_id(ids.EVIDENCE)
    allowed = frozenset({ids.MESSAGE, ids.EVIDENCE})
    assert ids.validate_ref(evidence, allowed=allowed) == evidence
    with pytest.raises(ValueError, match="expected one of"):
        ids.validate_ref(ids.new_id(ids.SNAPSHOT), allowed=allowed)
