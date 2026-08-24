"""The `.ctx` package format.

The product artifact: everything a second agent needs to pick up work a first
agent was doing, in one file that can be read by a machine, read by a person, and
checked by either.

**One JSON document, not an archive.** A zip would let the package carry the
whole history cheaply, and it is rejected for that reason: *human-inspectable* is
a requirement of this format, and a container you have to unpack before you can
read it is not. A `.ctx` opens in an editor, diffs in a review, and survives
being pasted into a bug report. When it grows too large for that, the answer is
to reference the archive rather than to inline it — which is the same answer the
format already gives.

**It carries interpretation, not history.** State items, the evidence they rest
on, and a pointer back to the archive the whole thing was derived from. Embedding
the archive by default would make the package a second copy of the conversation
that immediately begins to diverge from the first, and the archive already exists
and is already verifiable.

**Everything asserted can be checked.** The package hashes its own content, state
items name their sources, and the archive reference carries per-record hashes, so
a receiver holding the archive can verify the package describes *that* archive
and not a similar one. A receiver without the archive can still verify the
package is internally intact and knows exactly what it is missing.

**Versioned from the first byte.** ``format`` and ``format_version`` are the two
fields a reader must be able to trust before it understands anything else, so
they are required, and a package from an unknown future version is refused rather
than parsed optimistically.
"""

from __future__ import annotations

import hashlib
from typing import Any, Final, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from open_context.archive.records import canonical_json
from open_context.models.base import Timestamp, utc_now
from open_context.models.state import StateItem
from open_context.storage.schema import SCHEMA_VERSION

FORMAT: Final = "open-context-ctx"
"""What this file is. Present so a reader can reject a JSON file that is not one."""

FORMAT_VERSION: Final = 1
"""Incremented when a reader of the previous version could misread this one.

A package declaring a version above what a reader knows is refused. Reading it
anyway would mean guessing which fields changed meaning, and the failure of that
guess is silent.
"""

STATE_SCHEMA_VERSION: Final = SCHEMA_VERSION
"""The state model these packages carry.

Re-exported from storage rather than declared again, so the number in a package
is the same number the database migrated to and the two cannot drift.
"""

HASH_FIELD: Final = "content_hash"
"""Excluded from the hash it holds, by name, in one place."""


class Manifest(BaseModel):
    """What this package is, where it came from, and what can read it.

    Separate from the content so that identifying a package never requires
    parsing its state. A tool listing a directory of packages reads manifests.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    format: str = FORMAT
    format_version: int = Field(default=FORMAT_VERSION, ge=1)

    state_schema_version: int = Field(
        default=STATE_SCHEMA_VERSION,
        ge=1,
        description=(
            "Which shape the state items inside conform to. Separate from "
            "format_version: the envelope and its contents can change "
            "independently, and a reader that could not tell them apart would "
            "report a state model it does not understand as a corrupt package."
        ),
    )

    package_id: str = Field(min_length=1, description="Identity of this package.")
    session_id: str = Field(min_length=1, description="The session it was built from.")

    title: str | None = None
    source: str | None = Field(
        default=None, description="Where the session came from, such as 'chatgpt'."
    )

    created_at: Timestamp = Field(default_factory=utc_now)
    created_by: str = Field(
        default="",
        description=(
            "What produced this package — a tool name and version. Free text, "
            "because the honest answer is often 'a script somebody wrote'."
        ),
    )

    generator_version: str = Field(
        default="",
        description="Version of the code that built it, when it knows its own version.",
    )

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("format")
    @classmethod
    def _check_format(cls, value: str) -> str:
        if value != FORMAT:
            raise ValueError(f"not a {FORMAT} package: format is {value!r}")
        return value

    @property
    def readable(self) -> bool:
        """Whether this version is one this code understands.

        Older is fine — the format only adds. Newer is not, and is refused
        rather than parsed on the assumption that nothing important moved.
        """
        return self.format_version <= FORMAT_VERSION

    @property
    def state_readable(self) -> bool:
        """Whether the state items inside are a shape this code knows.

        Checked apart from ``readable`` so the two failures can be told apart. A
        package in a format this code reads, holding state it does not, is a
        specific and fixable situation; reporting it as an unreadable package
        would send a reader looking in the wrong place.
        """
        return self.state_schema_version <= STATE_SCHEMA_VERSION


class ArchiveReference(BaseModel):
    """A pointer to the archive this package was derived from.

    **Not a copy of it.** The reference carries the archive's identity, the range
    of records the package draws on, and a hash per referenced record. That is
    enough for a receiver holding the archive to prove the package describes that
    archive, and enough for a receiver without it to say precisely what it would
    need to fetch.

    ``location`` is a hint and nothing more. A path on the machine that built the
    package is meaningless on the machine that reads it, so nothing resolves it
    automatically and no validation depends on it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    archive_format: str = Field(min_length=1)
    archive_format_version: int = Field(ge=1)

    session_id: str = Field(min_length=1)
    first_seq: int = Field(ge=0)
    last_seq: int = Field(ge=0)
    record_count: int = Field(ge=0)

    location: str = Field(
        default="",
        description="Where the archive was when the package was built. A hint, never resolved.",
    )

    record_hashes: dict[int, str] = Field(
        default_factory=dict,
        description=(
            "seq to record hash, for the records this package cites. Lets a "
            "holder of the archive verify the package describes that archive."
        ),
    )

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.record_count and self.last_seq < self.first_seq:
            raise ValueError(f"archive range is inverted: {self.first_seq}..{self.last_seq}")
        outside = [seq for seq in self.record_hashes if not self.first_seq <= seq <= self.last_seq]
        if outside:
            raise ValueError(f"record hashes outside the stated range: {sorted(outside)[:5]}")
        return self


class EvidenceReference(BaseModel):
    """One piece of support for a state item, resolvable or not.

    **Provenance that cannot be followed is still worth carrying**, and saying so
    is the point of ``resolvable``. A state item extracted from a conversation
    knows which archive records it came from; whether the reader can reach those
    records depends on whether they have the archive. Dropping the reference
    because it might not resolve would destroy the only trail back to the source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_id: str = Field(min_length=1)
    archive_seqs: tuple[int, ...] = ()
    archive_record_ids: tuple[str, ...] = ()
    excerpt: str = Field(
        default="",
        description=(
            "A short quotation from the source, when one was captured. Present so "
            "a package without its archive can still show a reader what an item "
            "rests on, rather than only asserting that something does."
        ),
    )

    @property
    def resolvable(self) -> bool:
        """Whether this reference names anything at all to resolve."""
        return bool(self.archive_seqs or self.archive_record_ids)


class RecentContext(BaseModel):
    """Verbatim recent turns, when they were worth carrying.

    Optional by design. State is the durable layer and is what the format is
    for; recent context is an aid for a reader picking up mid-thought, and a
    package that omits it is complete, not truncated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[dict[str, Any], ...] = ()
    from_seq: int | None = Field(default=None, ge=0)
    to_seq: int | None = Field(default=None, ge=0)
    reason: str = Field(
        default="",
        description="Why these turns were included, so a reader can judge the choice.",
    )


class CtxPackage(BaseModel):
    """A portable context package.

    Construct through ``open_context.package.build``, which fills the manifest
    and the hash. Parsing is ``CtxPackage.from_json``, which refuses a package it
    cannot read rather than reading it partially.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    manifest: Manifest
    state: tuple[StateItem, ...] = ()
    inherited: tuple[StateItem, ...] = ()
    """Project-level state established by *other* sessions in the same project.

    **Kept apart from ``state``, not merged into it.** A package is one session's
    account of its own work, and the builder refuses foreign state on the ground
    that mixing them would present one agent's work as another's. That guard is
    right, so inherited context gets its own field rather than an exemption:
    a reader can always tell what this session concluded from what it merely
    inherited, and the distinction survives into the compiled context.
    """
    evidence: tuple[EvidenceReference, ...] = ()
    recent: RecentContext | None = None
    archive: ArchiveReference | None = None

    content_hash: str = Field(
        default="",
        description=(
            "sha256 over every other field, canonically serialised. Filled on "
            "construction when absent, so a package always carries one."
        ),
    )

    def model_post_init(self, _context: object, /) -> None:
        if not self.content_hash:
            object.__setattr__(self, HASH_FIELD, self.expected_hash())

    # ------------------------------------------------------------------
    # Integrity

    def body(self) -> dict[str, Any]:
        """Everything the hash covers: the package without its hash."""
        return self.model_dump(mode="json", exclude={HASH_FIELD})

    def expected_hash(self) -> str:
        """What ``content_hash`` should be for this content.

        Over ``canonical_json``, the same deterministic serialisation the archive
        uses, so two packages with the same content hash identically on any
        machine regardless of key order or float formatting.
        """
        return hashlib.sha256(canonical_json(self.body()).encode("utf-8")).hexdigest()

    def intact(self) -> bool:
        """Whether the content still matches the hash it carries."""
        return self.content_hash == self.expected_hash()

    # ------------------------------------------------------------------
    # Serialisation

    def to_json(self, *, indent: int = 2) -> str:
        """Human-inspectable by default.

        Indented and key-sorted: the format's own requirement is that a person
        can read it, and a single-line JSON document 40,000 characters wide meets
        the letter of that and none of the intent. ``canonical_json`` is still
        what the *hash* is computed over, so readability costs nothing.
        """
        import json

        return json.dumps(self.model_dump(mode="json"), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> CtxPackage:
        """Parse, refusing anything this code cannot read correctly.

        Version is checked before the body is trusted. A package from a future
        version may look parseable while a field has quietly changed meaning,
        and that is the failure this check exists to prevent.
        """
        import json

        document = json.loads(text)
        if not isinstance(document, dict):
            raise ValueError("a .ctx package must be a JSON object")

        manifest = document.get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError("a .ctx package must have a manifest")

        declared = manifest.get("format")
        if declared != FORMAT:
            raise ValueError(f"not a {FORMAT} package: format is {declared!r}")

        version = manifest.get("format_version")
        if isinstance(version, int) and version > FORMAT_VERSION:
            raise ValueError(
                f"package is format version {version}; this code reads up to "
                f"{FORMAT_VERSION}. Refusing to parse it rather than guess which "
                f"fields changed meaning."
            )
        return cls.model_validate(document)


__all__ = [
    "FORMAT",
    "FORMAT_VERSION",
    "STATE_SCHEMA_VERSION",
    "ArchiveReference",
    "CtxPackage",
    "EvidenceReference",
    "Manifest",
    "RecentContext",
]
