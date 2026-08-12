# Data model

Phase 1. Pydantic models only. No storage, no LLM calls, no persistence.

## Two rules

**Every record is frozen.** A changing fact produces a new record pointing at the old one through `supersedes`. Nothing is edited in place. The runtime has to answer "what did we believe in March, and why do we believe something else now", and in-place mutation destroys the first half of that question.

**Timestamps are timezone-aware UTC.** Naive datetimes raise. Aware ones are normalised to UTC on construction.

## Hierarchy

```
Record                          frozen, extra="forbid"
├── Session
├── Message                     seq, role, content, trust, parent_id
├── Evidence                    immutable excerpt, sha256 over NFC-normalised content
├── TokenEstimate               value + method + exact flag
├── ContextSnapshot             ids only, never content
└── StateItemBase               discriminated union on `type`
    ├── Goal                    goal_
    ├── Constraint              con_      + hard
    ├── Fact                    fact_     + subject
    ├── Decision                dec_      + rationale (required), alternatives
    ├── Task                    task_     + task_status, blocked_by
    ├── OpenQuestion            oq_
    ├── Preference              pref_
    ├── Entity                  ent_      + name (required), kind
    ├── Event                   evt_      + occurred_at
    └── Artifact                art_      + path, media_type
```

## Identifiers

Every id carries a type prefix: `msg_`, `ses_`, `ev_`, `snap_`, plus one per state type. A reference is therefore self-describing, and provenance is a flat list of strings rather than a list of (kind, id) pairs. A message id pasted where an evidence id belongs fails validation instead of dangling until something dereferences it.

Ids are random, not ordered. Ordering comes from explicit fields, never from an id.

## Decisions taken in Phase 1

**Discriminated union, not a flat type field.** A `Decision` without a rationale fails on construction rather than reaching the package builder and being presented to a model as settled.

**`Decision` and `Task` are union members, not parallel models.** The PRD defines both `StateItem(type="decision")` and a standalone `Decision`. Two definitions of the same thing drift.

**No `rejected_decision` type.** A rejected decision is a `Decision` with status `REJECTED`, which keeps the rationale and the alternatives attached to what they explain.

**`Message.seq` is explicit.** Timestamps are not a reliable ordering key. Provider exports give many messages the same second and some give none at all.

**References are validated for format only.** Phase 1 has no storage, so nothing can check that `msg_abc` exists. Existence checks arrive with the repository in Phase 2.

**`TokenEstimate` carries its method.** A character heuristic and a provider tokeniser disagree, and a budget spent on the wrong one overflows the window.

## Validation rules worth knowing

- A critical state item requires at least one source. An unsourced critical claim is the exact failure the runtime exists to prevent.
- `Decision.rationale` is required and non-empty.
- A tool message requires `tool_name`; every other role forbids it.
- A blocked task must record what blocks it.
- A snapshot cannot list the same message as both active and archived.
- Nothing can supersede, parent, or relate to itself.
- Evidence content cannot be empty, and a supplied `content_hash` that disagrees with the content raises.
- Message content **can** be empty. Real exports contain empty assistant turns, and rejecting them would make a genuine transcript unimportable.

## Provenance layering

```
message  ->  evidence  ->  state item
```

A message is what was said. Evidence is an immutable, hashed excerpt of it. A state item is an interpretation and cites the evidence, or cites the message directly when there is no excerpt worth materialising.

Evidence may also cite an artifact or event state item, which is the one place the chain passes back through state. It terminates rather than recursing, because a provenance reference must resolve to a record that is already stored. That single rule makes the graph acyclic by construction, so `state -> evidence -> state -> evidence -> ...` cannot be built at any depth. No schema change was needed to get this; it falls out of write ordering.

The rule differs from `supersedes` and `related_ids`, which are satisfied by a record appearing later in the same batch. Provenance is not.

## Session locality

Every reference resolves within one session: `Message.parent_id`, `Evidence.source_ids`, `StateItem.sources`, `supersedes`, `related_ids`, `Task.blocked_by`, and all snapshot membership. A cross-session reference raises `CrossSessionReferenceError`.

This is a property of the session state graph. A cross-session memory layer is no longer on the roadmap: the product is session handoff, and knowledge that should move between sessions moves as a portable context package rather than as a reference reaching sideways out of one session's history. Whether an imported package becomes a new session or a continuation of the original is unresolved; see [product-architecture.md](product-architecture.md).

## Known limitations

Records are frozen but not hashable, because `metadata` is a dict. Key collections by `id`. There is a test recording this so it is not rediscovered later.

`metadata` is typed `dict[str, Any]`, so a value such as a set passes validation and only fails when written to storage. The storage layer raises `SerializationError` naming the field and the record. Narrowing the type is a Phase 1 model change and has not been made.

`StateItemBase` has no `type` attribute; only the union members do. Code that needs the discriminator should be typed against `StateItem`, not the base class.

Cross-record referential integrity is enforced from Phase 2 onward. See [storage.md](storage.md).

`related_ids` may form a mutual pair: two items can each list the other. That is an association rather than a derivation, so it carries no ordering and no cycle to resolve. `supersedes` is different and is checked for cycles.
