# Incremental compaction

V2.2. Extract what arrived since last time instead of re-reading the whole
conversation.

## The problem it fixes

Phase 7 built watermarked incremental extraction — `Watermark`,
`extract_incremental`, `reconcile` — and **nothing called it**. Every package
re-read the whole archive and paid a model for the whole conversation again, so a
session repackaged ten times paid for its first turn ten times.

The missing half was somewhere to keep the watermark between runs. That is
`incremental_store`: state and watermark as one JSON file beside the archive.

## Measured

A session grown to 200 turns in batches of 40, repackaged after each batch —
which is what periodic capture actually looks like:

| | calls | prompt characters | latency | final state |
| --- | --- | --- | --- | --- |
| full re-read | 5 | 199,110 | 15.4 ms | 3 items |
| incremental | 5 | **75,802** | **4.6 ms** | 3 items |
| saving | — | **62%** | **70%** | none lost |

The shape matters more than the percentage. The incremental prompt stays flat at
about 15,000 characters as the session grows; the full read climbs from 14,746 to
64,926. **Cost tracks new events, not total ones**, so the saving widens with the
length of the session — which is the case the project exists for.

**One honest caveat.** These numbers come from a counting stand-in, not a real
model. That is the right instrument for *how much work is sent*, which is a
property of the caller and is measured exactly rather than through sampling
noise. It says nothing about whether a real model finds the same things from a
partial view — the pipeline preserves what it is given, and only a real run
answers the rest.

## Why the prompt is not just the new events

An incremental pass renders the state it already knows into the prompt as well.
That is deliberate and Phase 7 paid for the lesson: a model shown only new
messages has never seen the decision being overturned, so a reversal it reports
resolves to nothing. The first real incremental run produced exactly that — two
orphaned assertions beside a still-active predecessor.

So the prompt is *new events plus known state*, which is why a second run is not
smaller than the first. It is smaller than re-reading everything, and stays that
size while the conversation does not.

## What the watermark promises

**It advances only on success.** An extraction whose reply was malformed, or
whose model was rate-limited, leaves it where it was, so the unread range is read
next time rather than skipped permanently and silently.

**It is stored with the state that earned it.** Written apart, a crash could
leave a watermark claiming events had been read beside state that never saw them,
and nothing downstream could detect the gap. They are replaced together through
an atomic rename.

**Damage reads as "nothing known".** A truncated file means the previous run died
mid-write; reading the session again is recovery, and refusing to run would turn
an interruption into a permanent failure. A file naming a different session is
ignored for a sharper reason: adopting it would attribute one conversation's
decisions to another.

**Conflicts block rather than merge.** When new state contradicts what is stored,
the update is reported and the existing state is kept. Silently preferring the
newer item is a policy nobody chose, and is indistinguishable from having found
no conflict.

## Using it

On by default:

```python
items, warnings = extract_state(archive_root, session_id, provider)
```

`incremental=False` forces a full re-read — useful when the stored state is
suspect, or to reproduce a measurement.
