"""One real request per configured provider.

**Skipped unless the credential is in the environment**, so ordinary runs and CI
never touch the network. Marked ``llm``, which the project's pytest
configuration already describes as "requires a configured LLM provider and is
skipped by default".

Run one deliberately:

```
GROQ_API_KEY=... pytest -m llm tests/integration/test_real_provider_smoke.py
OPENROUTER_API_KEY=... OPENROUTER_MODEL=... pytest -m llm
```

Nothing here prints a key, and the assertions check that no result or error
carries one.

**Only a rate limit is treated as transient.** The provider layer already sorts
failures into categories, and this test relies on that: a 429 skips with its
reason, while an authentication failure, a vanished model, a malformed response,
or a refused request all fail. Collapsing them would let a broken provider look
like a busy one.
"""

import os

import pytest

from open_context.llm import (
    AuthenticationError,
    ChatMessage,
    FinishReason,
    GenerationRequest,
    ModelUnavailableError,
    ProviderConfig,
    RateLimitError,
    create_provider,
)
from open_context.llm.free_models import (
    BillingClass,
    approved_models,
    billing_class_of,
    find_free_model,
)
from open_context.llm.providers import groq, openrouter

pytestmark = [pytest.mark.llm, pytest.mark.integration]

ANSWER_TOKENS = 120
"""Room for the answer itself, on top of whatever the model spends thinking."""


def prompt_for(spec) -> GenerationRequest:
    """A request whose output cap suits *this* model.

    **Derived, never hardcoded.** The cap used to be a flat 300, which was fine
    until every approved Groq model turned out to reason: 300 sits inside their
    512-token thinking floor, so ``check_output_cap`` refused the request before
    it was sent and the test failed on every run where a credential existed.

    Asking the registry keeps the test and the gate agreeing by construction —
    a model whose floor changes moves this with it.
    """
    return GenerationRequest.of(
        ChatMessage.system("Answer in one short sentence."),
        ChatMessage.user("Name one reason to write tests."),
        max_output_tokens=spec.reasoning_overhead + ANSWER_TOKENS,
        temperature=0.0,
    )


@pytest.mark.parametrize(
    ("name", "env_var", "model_var"),
    [
        ("groq", groq.API_KEY_ENV, "GROQ_MODEL"),
        ("openrouter", openrouter.API_KEY_ENV, "OPENROUTER_MODEL"),
    ],
)
def test_one_real_request(name, env_var, model_var):
    """A single generation, end to end, through the ordinary interface.

    **Free models only.** The model comes from the allowlist, and an override
    must also be on it. A provider with no approved model — Groq, today — skips
    rather than reaching for something priced.
    """
    key = os.environ.get(env_var)
    if not key:
        pytest.skip(f"{env_var} is not set")

    approved = approved_models(name)
    if not approved:
        pytest.skip(f"no {name} model is approved as free; see open_context.llm.free_models")

    model = os.environ.get(model_var, approved[0].model_id)
    if find_free_model(name, model) is None:
        pytest.skip(f"{model} is not on the free allowlist")

    spec = find_free_model(name, model)
    assert spec is not None

    provider = create_provider(ProviderConfig(provider=name, model=model))
    try:
        result = provider.generate(prompt_for(spec))
    except RateLimitError as exc:
        # Transient, and only this. A shared free tier throttles without warning,
        # and failing the suite over it would make a green run depend on somebody
        # else's traffic. Every other category below is a real defect and is
        # allowed to fail loudly.
        pytest.skip(f"{name} is rate limiting, which is transient: {exc}")
    except AuthenticationError:
        raise
    except ModelUnavailableError:
        raise

    assert result.provider == name
    assert billing_class_of(name, model) is BillingClass.FREE, "only free models may be called"
    assert result.text.strip() or result.finish_reason is FinishReason.LENGTH, (
        "the model said something, or ran out of output budget saying it — a reasoning "
        "model can spend its whole budget before emitting content, which is why the "
        "budget above is not tiny"
    )
    assert result.model, "the served model identifier was recorded"
    assert isinstance(result.finish_reason, FinishReason)

    if result.usage is not None:
        assert result.usage.input_tokens is None or result.usage.input_tokens > 0
        assert result.usage.output_tokens is None or result.usage.output_tokens > 0

    serialized = result.model_dump_json()
    assert key not in serialized, "no credential reached the result"
    assert key not in repr(provider)
