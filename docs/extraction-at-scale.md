# Extracting a real session

The step that turns a conversation into goals, constraints, and decisions is the
only one that needs a model — and the only one that met a wall on real data.

## What broke

The first live extraction against a real transcript:

```
extraction failed (ProviderUnavailableError): groq returned an unexpected 413:
Request too large for model `openai/gpt-oss-120b` ... on tokens per minute (TPM):
Limit 8000, Requested 19178
```

One-shot extraction sends the whole session in a single request. That works on
the sessions a test suite builds and on nothing a person actually has: 19,178
tokens of conversation against an 8,000-token-per-minute ceiling.

## Windows

The session is read in pieces small enough to send, each carrying the state found
so far:

```
window 0-12   ->  3 facts
window 12-25  ->  1 constraint      (given the 3 facts)
window 25-38  ->  1 decision        (given all 4)
```

**Carrying state forward is what makes windows safe.** A window that saw only its
own messages has never seen the decision being overturned in a later one, and
would report a supersession resolving to nothing — exactly what Phase 7 measured
before `known_state` existed. That carried state is also why windows are sized
generously rather than as small as possible.

**Windows are cut on record boundaries.** Half a tool result is not a smaller
tool result; it is a different one, and a model asked to read it would extract a
fact the conversation never contained. A record larger than a whole window gets a
window to itself — splitting it would be worse than sending something too big,
which at least fails loudly.

## Rate limits are waited out

The second live run failed 32 of 35 windows, while the provider was naming the
exact number of seconds that would have fixed each one:

```
Rate limit reached ... Please try again in 11.43s
```

`RateLimitError.retry_after` has been carried since Phase 4.5 and deliberately
never acted on — the provider boundary refuses to choose a retry policy for its
callers. Windowed extraction *is* a caller, and is where the policy belongs.

With pacing, the same session went from **3 items and 32 failed windows** to
**19 items and 11 failed windows**.

Bounded on both sides: three attempts per window, and nothing waits longer than
30 seconds for one. A provider asking for ten minutes is telling you to come back
later, and a tool that silently obeyed would look like it had hung. Time spent
waiting is reported, so a slow run is explicable rather than mysterious.

## A failing window does not fail the session

Rate limits are transient and common. Losing four windows of state because the
fifth was throttled would make a long session strictly *harder* to extract than a
short one.

So failures are counted and named, the state that was gathered is kept, and
`complete` says whether any window was missed:

```
warning: 11 of 35 windows could not be read; the state below is real but
         incomplete
```

That distinction matters because the incompleteness is invisible in the state
itself. Nine correct facts look exactly like nine correct facts whether or not
there were twelve.

**The watermark does not advance on a partial read.** An incomplete extraction
must not claim the range was covered, or the unread windows are skipped
permanently and silently.

## Provenance survives

Phase 6 put archive record ids on the `ExtractedItem` *wrapper* rather than on
`sources`, because the two id schemes had never been reconciled. Anything that
unwraps the item to get a `StateItem` therefore loses the trail — which the first
windowed implementation did, producing packages where every item read:

```
warning: 19 of 19 state items have no archive provenance
```

`WindowedResult.provenance` carries the mapping alongside, so a package built
from a live extraction now reports `traced 9 of 9` and validates at provenance
depth.

## Using it

```bash
export OPEN_CONTEXT_PROVIDER=groq
export OPEN_CONTEXT_MODEL=openai/gpt-oss-120b
export OPEN_CONTEXT_API_KEY=...

open-context handoff session.jsonl --store ./ctx --out project.ctx --extract
```

Every approved Groq model reasons before it answers as of 2026-08-23, so the
output cap must leave room for thinking — `check_output_cap` refuses a cap that
does not, before the request is sent. See [llm.md](llm.md).
