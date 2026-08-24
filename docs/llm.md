# LLM provider and tokenizer

Phase 4.5, extended in Phase 5.7. The boundary that keeps model providers replaceable, and the first implementations behind it.

```
future compaction engine
        |
LLMProvider / Tokenizer          <- this phase
        |
one provider implementation
        |
a model
```

Open Context has a provider-neutral interface through which model implementations are used, along with token counting and model capability abstractions.

Phase 4.5 built the boundary and shipped deterministic fakes only. Phase 5.7 added the first real implementations — Groq and OpenRouter — behind it, without changing the interface.

## What this phase is not

No compaction, no summarization, no extraction, no retention, no retrieval, no ranking, no message selection, no package format, no context-pressure detection. The boundary builds the thing they call, and nothing that calls it.

## The dependency rule

```
archive -> state -> future compaction -> LLMProvider / Tokenizer
```

and never

```
archive -> a vendor SDK
```

A compaction engine written against these interfaces contains no branch on which provider answered. `if provider == "openai"` above the boundary means the abstraction has failed.

The rule is enforced by tests that read the source (`tests/unit/test_llm_boundary.py`), not by convention:

| Checked | Why |
| --- | --- |
| `archive`, `models`, `storage`, `importers` never import `open_context.llm` | Nothing below compaction needs a model, and one import would make those layers untestable without one |
| No package imports a vendor SDK | A provider implementation is the only place one may appear |
| `open_context.llm` never imports the rest of the project | It is a boundary, not a participant. It can be lifted out and tested alone |
| The `llm` **abstraction** never imports a network or subprocess module | Why the suite needs no internet, no key, and no model server. `llm/providers` is excluded, since reaching a service is what a provider is for — and a separate test asserts that package is the only network code |
| The abstraction never imports its own providers | Otherwise wanting the interface would drag in an HTTP client |
| All of `open_context.llm` uses only the standard library and pydantic | The base package stays lightweight. No SDK was added for either provider |

## LLMProvider

```python
class LLMProvider(Protocol):
    def model_info(self) -> ModelInfo: ...
    def generate(self, request: GenerationRequest) -> GenerationResult: ...
    def structured_output(self, request, schema: Mapping[str, Any]) -> StructuredResult: ...
```

Three methods. Base URLs, auth headers, SDK objects, and wire formats live inside an implementation and are invisible above it.

`model_info()` is a method rather than an attribute because a provider may have to ask the service what it is talking to, and may cache the answer.

`GenerationRequest` carries only knobs that mean the same thing everywhere — messages, `max_output_tokens`, `temperature`, `stop`. Anything provider-specific goes in `options`, where it is visibly the caller's decision to couple to one provider rather than something this interface blessed.

`GenerationResult` carries the text, which model answered, a `FinishReason`, optional provider-reported `usage`, and `raw`, the provider response as parsed with unknown fields preserved — the same discipline the import layer applies to conversation records.

### Usage is not an estimate

`GenerationResult.usage` is the provider's own accounting, which is authoritative: it is what a bill is computed from. It is `None` when the provider said nothing, and it is never filled in from a `Tokenizer`. A number that might be measured or might be guessed is worse than a missing one, which is why `TokenUsage` and `TokenCount` are separate types that cannot be confused for one another.

## Tokenizer

```python
class Tokenizer(Protocol):
    def model_info(self) -> ModelInfo: ...
    def count_text(self, text: str) -> TokenCount: ...
    def count_messages(self, messages: Sequence[ChatMessage]) -> TokenCount: ...
```

Separate from `LLMProvider` because token counting is a genuinely different problem. A count may be local or remote, model-specific or generic, exact or approximate, or simply unavailable — and not every provider that generates text can count tokens for what it generates.

### Exact versus estimated

**The invariant: the system never presents an estimate as an exact token count.**

```python
TokenCount(count=1234, exact=True, method="cl100k_base")
TokenCount(count=1180, exact=False, method="chars/4")
```

A compactor that fills a 128,000 token window to 127,900 on the strength of a character heuristic will overflow, and the failure will look like a provider bug rather than a measurement one. Knowing a number is approximate is what lets a caller leave headroom.

The invariant is enforced by arithmetic rather than by discipline:

```python
exact + estimate  # -> exact=False, method="mixed"
```

A total is only as trustworthy as its worst part, and there is no way to add your way back to `exact=True`. `method` is for reporting; `exact` is what code branches on.

### Messages cost more than their text

A chat model wraps each message in role markers and delimiters defined by its own template. A tokenizer that does not know that template cannot count the framing exactly, so `count_messages` may be estimated where `count_text` is exact. Saying so is more useful than a number quietly short by a few tokens per turn.

### The three states

| State | Implementation | Behaviour |
| --- | --- | --- |
| Exact | A real tokenizer for the model | `exact=True`, `method` names the encoding |
| Estimated | `CharacterRatioTokenizer` | `exact=False` always, `method="chars/4"` |
| Unavailable | `UnavailableTokenizer` | Raises `TokenizationUnavailableError`, with a reason |

`UnavailableTokenizer` exists so "we cannot count this" is a real object that can be passed around and handled, rather than a `None` every call site has to remember to check. It still reports `model_info()`, so a caller can name the context window it cannot yet fill.

`CharacterRatioTokenizer` is a deliberately crude fallback. The ratio is an English-prose rule of thumb: it runs long on code, punctuation, and non-Latin scripts, and it knows nothing about any chat template. It is offered because a rough number a caller knows is rough beats no number at all.

## ModelInfo and capabilities

```python
ModelInfo(
    provider="ollama",
    model="llama3.1",
    context_window=131_072,
    max_output_tokens=4_096,
    tokenizer="llama3",
    capabilities=frozenset({Capability.TEXT_GENERATION, ...}),
)
```

| Capability | Meaning |
| --- | --- |
| `TEXT_GENERATION` | Can produce text |
| `STRUCTURED_OUTPUT` | Can produce data conforming to a schema |
| `TOOL_CALLING` | Can call tools |
| `VISION` | Accepts images |
| `EXACT_TOKEN_COUNT` | An exact count is obtainable |

Only capabilities this project will act on. A flag nothing branches on is a claim nobody verifies.

**Unknown is a value.** `context_window` and `max_output_tokens` are `None` when the provider did not say. Filling in a plausible 128,000 would be worse than admitting ignorance: a budget computed from an invented window looks authoritative and is wrong, and the failure surfaces much later as a truncated request nobody can explain.

**An unsupported capability raises.** `UnsupportedCapabilityError`, rather than degrading to "ask nicely and hope for JSON". A silent fallback would make a capability flag a suggestion and mislead anyone who checked it.

**No pricing.** Deliberately. Prices change without notice, vary by region and contract, and are not provider-neutral in any useful way, so a table here would be stale before it was read. This phase is about token and context accounting; billing is separable and is not in scope.

## Structured output

```python
result = provider.structured_output(request, schema_for(MyModel))
result.data  # parsed
result.text  # what the model actually emitted
```

Providers reach structured output by different routes — a native structured mode, a JSON mode, grammar-constrained decoding, or a prompt that asks politely. The route is the provider's business. What crosses the boundary is the same either way: parsed data, or `MalformedStructuredOutputError` carrying what the model actually said.

JSON Schema is the neutral schema language, not because it is pleasant but because every provider that supports structured output accepts it and pydantic emits it. `schema_for(model)` converts a pydantic model at the edge, so nothing downstream learns pydantic was involved.

**How far validation goes, exactly.** `parse_structured_output` strips a markdown fence, parses JSON, and checks the top-level `type` and the presence of `required` properties. That catches prose instead of JSON, a fenced block, and an object missing half its fields. It is **not** full JSON Schema validation: nested constraints, formats, enums, and value types are not checked. A caller needing real validation validates the data itself, which with pydantic is one `model_validate` call. The limit is pinned by a test so it stays a documented boundary rather than an assumption.

No prompts and no domain schemas live here. Building those is a later phase's work.

## Errors

Provider-neutral, all under `LLMError`.

| Error | Meaning |
| --- | --- |
| `ProviderUnavailableError` | Not reachable, not running, or not registered |
| `AuthenticationError` | Credentials missing, malformed, or rejected |
| `ModelUnavailableError` | Provider is reachable but this model is not on it |
| `InvalidRequestError` | Rejected as malformed or unsatisfiable |
| `RateLimitError` | Refusing further requests for now; carries `retry_after` when given |
| `ContextLimitExceededError` | Exceeded the context window; carries counts when the provider gave them |
| `UnsupportedCapabilityError` | The model cannot do what was asked |
| `MalformedStructuredOutputError` | Not the structure that was asked for; carries the text |
| `EmptyCompletionError` | Cut off by the output cap before writing anything; carries the reasoning spend |
| `TokenizationUnavailableError` | No tokenizer for this model |

**A vendor exception must never reach the core.** A provider implementation translates whatever its SDK or HTTP layer raises into one of these, so code above the boundary handles "rate limited" without importing anyone's exception classes or matching on message strings. A provider that lets its own exception escape has not finished its job.

**Translation is not swallowing.** The original belongs on `__cause__` via `raise ... from exc`. Categories exist so a caller can decide, not so detail can be discarded.

**No retries.** No backoff, no attempt budget, no circuit breaker. `RateLimitError.retry_after` carries the information a policy would need; the policy waits until something actually calls this in a loop.

## Reasoning models spend the output cap before they answer

A reasoning model thinks before it writes, and that thinking is billed against
`max_output_tokens`. Set the cap below what it spends and the provider returns
**HTTP 200, a well-formed body, and nothing in it**.

Every approved model, called to find out. Groq measured 2026-08-23, OpenRouter
2026-08-24:

| Model | Reasoning | Observed |
| --- | --- | --- |
| `google/gemma-4-31b-it:free` | none | answers directly, 0 reasoning tokens |
| `google/gemma-4-26b-a4b-it:free` | none | answers directly, 0 reasoning tokens |
| `nvidia/nemotron-3-nano-30b-a3b:free` | separate field, counted | 44 reasoning tokens on a trivial prompt |
| `nvidia/nemotron-3-super-120b-a12b:free` | separate field, counted | 91 reasoning tokens |
| `liquid/lfm-2.5-2.6b:free` | separate field, counted | 78 reasoning tokens |
| `openai/gpt-oss-20b` | separate field, counted | at a 160-token cap: **158 tokens of thinking, empty `content`** |
| `openai/gpt-oss-120b` | separate field, counted | same shape, 50 tokens on a trivial prompt |
| `qwen/qwen3.6-27b` | inline `<think>`, **uncounted** | 158 tokens inside an unterminated `<think>` |

**Three of those OpenRouter entries used to say `none`, and nobody had checked.**
They were written down as direct-answering because that was the default, so
`check_output_cap` had been silently inert for that provider since the day it was
added. Calling them took one request each.

**Returning that as an empty string is what makes it dangerous.** An empty
summary is not obviously wrong to anything downstream: it is stored, passed to a
judge, and scored. A Phase 8.5 benchmark run produced 26 cells this way and read
as the clearest result the project had — every compaction arm near zero, the
reference arm at 1.00. The reference arm is the one that makes no compaction
call. Written up in [adversarial.md](adversarial.md); the cells themselves sit
in the git-ignored `benchmarks/results/void/`.

Four things now stand between that and a result:

- **`EmptyCompletionError`** — empty content with `finish_reason: length` is an
  error. A *truncated* answer is still an answer and is returned; losing the
  whole response is the failure.
- **`TokenUsage.reasoning_tokens`** — broken out of `output_tokens`, not added to
  it, because it is the part of the allowance the caller never receives.
- **`strip_inline_reasoning`** — a leading `<think>` block is removed from
  `content`. The inline kind is the harder one: nothing in the accounting says it
  happened, and unstripped it is deliberation graded as a conclusion. An
  unterminated block leaves an empty string, which the check above then catches.
- **`check_output_cap`** — refuses a cap inside the model's thinking budget
  *before* the request is sent, and names the models that answer directly.

### A value is a measurement, or it is `UNKNOWN`

The registry records `Reasoning` from a live call, never from a model card or a
name, and every entry carries `reasoning_measured_on` — the date it was seen. A
value without a date does not count as measured, and `reasoning_measured` says so.

**An unmeasured entry defaults to `UNKNOWN` and is given the full reasoning
floor.** The failure modes are not symmetrical: assuming a direct model reasons
costs some output budget, while assuming a reasoning model answers directly
returns an empty completion. The default used to be `NONE` — the one value that
turns the check off — which is exactly how five entries came to claim they did
not reason without anyone having called them.

The floor is a single number rather than a per-model figure. The observed cost
ranges from 44 to 332 tokens depending on the prompt, so a per-model value copied
from one observation would be false precision; the question the gate answers is
not "how much does it usually spend" but "is this cap so small that no answer can
survive it".

## Configuration

```python
config = from_env()  # OPEN_CONTEXT_PROVIDER, _MODEL, _BASE_URL, _API_KEY
provider = create_provider(config)
```

A name and a model, resolved through a registry to an implementation. Environment variables and a frozen dataclass; no config server, no secrets manager, no file format to version. A local-first tool that needed infrastructure to decide which model to call would have lost the plot.

`from_env` takes an optional mapping, so tests never touch the real environment.

**Credentials.** `api_key` is read from the environment and excluded from `repr`, so it cannot leak into a log line, a traceback, or an assertion dump. Nothing in this repository contains a key and nothing should; the field carries one from the environment to a provider and no further. Pinned by a test.

## Providers and optional dependencies

```python
register_provider("ollama", build_ollama)  # called when the module is imported
create_provider(ProviderConfig(provider="ollama", model="llama3.1"))
```

The registry is what keeps vendor SDKs out of the base package. A provider registers itself on import, so a module nobody imports costs nothing. Installing an extra makes a provider available; not installing it makes that name unknown, which surfaces as `ProviderUnavailableError` listing what *is* available — an ordinary, explainable failure rather than an `ImportError` from the middle of a call stack.

| Provider | Credential | Status |
| --- | --- | --- |
| `fake` | none | Deterministic fakes: `FakeProvider`, `FailingProvider`, `WordTokenizer` |
| `groq` | `GROQ_API_KEY` | Implemented; five models approved on `ACCOUNT_TIER` evidence — see below |
| `openrouter` | `OPENROUTER_API_KEY` | Implemented, with approved free models |
| Ollama, Anthropic | — | Not written |

```python
import open_context.llm.providers  # registers groq and openrouter

provider = create_provider(ProviderConfig(provider="groq", model="llama-3.1-8b-instant"))
```

**Providers are not imported by `open_context.llm`.** Importing the abstraction gets you the interface, the tokenizer contract, the errors, and the fakes — and no HTTP client. A caller that wants a real model imports `open_context.llm.providers` on purpose. A test enforces this, so a convenient import cannot quietly make every consumer of the interface network-dependent.

**No dependency was added.** Both post JSON over `urllib`. An SDK per vendor is a large surface for a small benefit, and the base package keeping its single runtime dependency is worth more. Transport is injectable, so provider tests exercise the real translation and error mapping against canned responses without opening a socket.

**Both are their own providers, not one generic OpenAI-compatible one.** They happen to share a wire dialect, and the translation for it lives in a private helper both call — shared implementation, not a second abstraction. Each owns its name, credential, endpoint, headers, and defaults.

## Free models only

**The initial real-model benchmark supports FREE MODELS ONLY.**

**The benchmark refuses to execute models that are not explicitly approved as free.**

**API-key access does not imply that a model is free.**

**No paid fallback exists.**

A credential opens a door; it says nothing about what is behind it. Groq's key reaches a dozen models that all cost money. OpenRouter's reaches hundreds, of which a handful are free. So the only authority is an explicit allowlist in `open_context.llm.free_models`, and every entry records the date its price was read as zero and the endpoint it was read from.

```python
FreeModelSpec(
    provider="openrouter",
    model_id="openai/gpt-oss-20b:free",
    billing_class=BillingClass.FREE,
    context_window=131_072,
    verified_on="2026-08-13",
    verified_by="openrouter /api/v1/models reported pricing.prompt == 0 and completion == 0",
)
```

### Validation happens before any request

```
configured model -> allowlist -> approved? -> yes -> HTTP
                                           -> no  -> ModelNotApprovedError, ZERO requests
```

The check runs in the provider constructor, **before the credential is even read**, so an unapproved model costs no network call and no money. A counting transport proves it: `test_an_unapproved_model_makes_no_http_request`.

A `:free` suffix is never consulted. It is a naming convention, conventions drift, and a model that stopped being free would keep its name. `foo:free` absent from the allowlist is refused exactly like `foo`.

### Two kinds of evidence, and they are not equal

Groq's `/v1/models` reports a positive `pricing.prompt` and `pricing.completion` for **every** chat model it serves — `llama-3.1-8b-instant` is priced at $0.00000005 per prompt token, re-checked 2026-08-14. Groq's free tier is an account-level allowance against priced models, not a set of zero-priced ones, and nothing in the API distinguishes "within my allowance" from "billed". No plan or billing field is exposed, and the models endpoint returns no rate-limit headers, so **it cannot be established from inside this repository.**

For a time `approved_models("groq")` was therefore empty. It was then established the only way it can be: the account owner confirmed their plan does not bill for these models. That is a different kind of fact from a published price, so the registry distinguishes them in the type rather than only in prose.

| Evidence | Means | Re-checkable by |
| --- | --- | --- |
| `ZERO_PRICE` | The catalogue reported a per-token price of exactly zero | Anyone, from the endpoint in `verified_by` |
| `ACCOUNT_TIER` | The account owner confirmed their plan does not bill for it | Only that account's owner |

**`ACCOUNT_TIER` entries are not portable.** A different account, or this one on a different plan, makes them false, and nothing here can detect that happening. Anyone reusing this repository should delete the Groq entries rather than inherit somebody else's billing arrangement.

Approving five Groq models did not approve the provider wholesale: everything else Groq serves is still refused, pinned by a test.

### What the free tiers actually give

Measured from `x-ratelimit-*` response headers on 2026-08-14, not from documentation:

| Provider / model | Requests/day | Token limit | Latency |
| --- | --- | --- | --- |
| OpenRouter `:free` models | **50** | — | 10–20s |
| Groq `llama-3.3-70b-versatile` | **1000** | 12k/min | ~0.3s |
| Groq `llama-3.1-8b-instant` | **14400** | 6k/min | ~0.3s |

Groq's allowance is twenty times OpenRouter's and roughly thirty times faster per request, which is what made a complete benchmark matrix possible in a single session rather than across three days. Groq's binding constraint is tokens per *minute*, so a run paces against that; OpenRouter's is requests per *day*, which pacing cannot help with.

### No paid fallback

A free model that is unavailable, rate-limited, over quota, timing out, or erroring **fails**. Nothing selects a different model, a different provider, or a paid tier. Choosing another approved free model is the caller's explicit act, never the code's — there is no fallback mechanism to review, and a test asserts no such machinery exists.

### The registry goes stale

Free tiers are withdrawn without notice; one model this project tried during Phase 5.7 had already stopped being free, and OpenRouter's daily free allowance is small enough to exhaust in a session. Re-verify before trusting an entry, and read a `ModelNotApprovedError` as the registry needing attention rather than as an obstacle to route around.

### Configuration

| Variable | Meaning |
| --- | --- |
| `GROQ_API_KEY` | Groq credential |
| `OPENROUTER_API_KEY` | OpenRouter credential |
| `OPENROUTER_REFERRER`, `OPENROUTER_TITLE` | Optional attribution, absent unless set |

Options accept `base_url`, `timeout`, `context_window`, and `max_output_tokens`.

**No model is hard-coded.** Catalogues and free tiers change without notice, so the model is always the caller's, and the result records the identifier the service reported — which may differ from the one asked for.

**No context window is invented.** Neither API returns one with a completion, so `ModelInfo.context_window` is whatever was configured explicitly and `None` otherwise. A plausible-looking guess would be a number every budget decision downstream would trust.

**A missing credential is an error.** Never a quiet fall back to another provider or to the fake — a real-model run that silently became a deterministic one would produce numbers labelled as measurements.

### Structured output is refused, not faked

Both `structured_output` implementations raise `UnsupportedCapabilityError`, and neither advertises `Capability.STRUCTURED_OUTPUT`. Both services expose a JSON mode, but whether a given model honours a schema varies — and OpenRouter routes to many models with different support — so a flag set here would be right for some routes and wrong for others. Prompting for JSON and calling it native structured output would make the capability a guess. The Phase 5.6 benchmark does not need it.

### What the providers send

Every generic field the interface defines is translated: messages (with `name`), `max_output_tokens` → `max_tokens`, `temperature`, `stop`. Nothing is silently discarded. `options` merges last, so a caller reaching for a provider-specific setting gets it and can see that they did.

Both send a `User-Agent`. Not decoration: Groq's edge answers an unidentified `urllib` request with a Cloudflare 403 before it reaches the API.

### Error translation

| Status | Becomes |
| --- | --- |
| 401, 403 | `AuthenticationError` |
| 429 | `RateLimitError`, with `retry_after` when the header gives one |
| 404 | `ModelUnavailableError` |
| 400, 422 | `InvalidRequestError`, or `ContextLimitExceededError` when the message says so |
| 5xx | `ProviderUnavailableError` |
| no response at all | `ProviderUnavailableError` |

A context overflow arrives as an ordinary 400 from both services, so it is recognised from the provider's own error text. Crude, and a miss degrades to `InvalidRequestError`, which is still true.

**No retries.** `retry_after` is carried, never acted on. Retry policy stays out of the provider boundary, as it has since Phase 4.5.

### Security

The credential travels in one place — the `Authorization` header — and nowhere else. It is not in the URL, not in the body, not in `repr`, not in any error message, and not in any serialized result; `HttpRequest.redacted_headers()` is the only sanctioned way to render a request. `.env`, `.env.*`, and `*.key` are git-ignored. No key appears in this repository, in a test fixture, or in a benchmark artifact, and tests use a placeholder string.

## Streaming — a future capability, not a current one

**Not in this interface, and not in the capability set.** There is no streaming method on `LLMProvider`, and `Capability` has no `STREAMING` member.

An earlier draft carried the flag without the method, on the reasoning that a provider might as well advertise what it can do. That was wrong in a small but specific way: a flag describing something the interface offers no way to call is a claim a caller can read and then do nothing with, and a capability set is only worth consulting if every member answers a question that changes what code does. So it is gone until there is a streaming call to guard.

Nothing in the planned architecture needs one. Compaction is a batch operation, and streaming matters for interactive display, which this project does not do. Designing the contract now, against zero verified provider implementations, would mean designing it twice.

When it returns it brings both halves together: a streaming method, and the flag that says whether a given model supports it.

## Testing

**The default run is deterministic and offline.** No API key, no internet, no Ollama, no model server, no external database. Provider tests included: transport is injectable, so the real request translation and error mapping are exercised against canned responses without opening a socket.

The one exception is opt-in. `tests/integration/test_real_provider_smoke.py` is marked `llm`, which the project's pytest configuration already describes as skipped by default, and skips again if the credential is absent from the environment. It sends a single generation through the ordinary interface and asserts that no credential reaches a result, a `repr`, or an error.

### Smoke test result, 2026-08-13

| Provider | Model | Outcome |
| --- | --- | --- |
| `openrouter` | `openai/gpt-oss-20b:free` | Generated |
| `openrouter` | `liquid/lfm-2.5-2.6b:free` | Generated |
| `openrouter` | `nvidia/nemotron-3-nano-30b-a3b:free` | Generated |
| `openrouter` | `google/gemma-4-26b-a4b-it:free` | Generated |
| `openrouter` | `nvidia/nemotron-3-super-120b-a12b:free` | Generated |
| `openrouter` | `google/gemma-4-31b-it:free` | 429, rate-limited |
| `groq` | — | Skipped: no approved free model |

All six models were re-confirmed at `pricing.prompt == 0` and `pricing.completion == 0` on the same date before any request was sent. The rate-limited model was recorded as such and not substituted; it is the allowlist's first entry and therefore the default, so a run that hits 429 immediately is expected and is the caller's cue to name another approved model explicitly.

## Unresolved

**Which providers offer reliable exact token counting?** Still unmeasured. Neither Groq nor OpenRouter ships a tokenizer, and both report usage only after a call — which is `TokenUsage`, not a `Tokenizer`. Neither advertises `EXACT_TOKEN_COUNT`, so a benchmark against them counts with an estimator and says so.

**Where do context windows come from?** Configured or unknown. OpenRouter publishes them through a models endpoint that the provider deliberately does not call, because one generation should be one request. Fetching and caching them is a reasonable future addition.

**How should multimodal tokens be counted?** An image is not characters, and its cost depends on resolution and the provider's tiling. `ChatMessage.content` is text only, so the question is deferred rather than answered badly.

**How should tool-call tokens be counted?** Tool definitions and call payloads occupy context and are serialised differently by every provider. Not represented here.

**How does provider-specific serialisation affect counts?** The same messages become different token counts under different chat templates. This is why `count_messages` exists separately from `count_text`, but the framing allowance is a guess in every tokenizer that does not know the template.

**Should tokenizers live beside providers or independently?** Currently independent, which is why a provider may hand back a tokenizer it did not implement. A local model whose tokenizer ships with its weights may argue for coupling them.

**Should provider adapters be separate distributions?** Optional dependency groups in one package are the plan, and one package is simpler while nothing has shipped. Separate distributions become worth it only if a provider needs a release cadence of its own.

**How should a model's context window be discovered, and kept current?** Providers change windows without renaming models. `ModelInfo` is a snapshot with no expiry, and `None` is the honest answer when a provider does not report one.

**How should structured-output differences be normalized?** The interface promises parsed data, but a provider with no native structured mode has to prompt for JSON, and its failure rate is different in kind. Whether that difference should be visible to a caller is unresolved.

**Should streaming be part of the stable interface?** Deferred, above. Neither the method nor the capability flag exists today.

**How should local model tokenizers be discovered?** A local model may ship a tokenizer alongside its weights, in a format that varies by runtime. Nothing here looks for one.
