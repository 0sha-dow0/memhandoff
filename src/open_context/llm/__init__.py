"""Provider-neutral LLM and tokenizer abstraction.

```
caller -> LLMProvider  -> a provider implementation -> a model
caller -> Tokenizer    -> a tokenizer implementation
```

The boundary that keeps model providers replaceable. Code above it asks for
text, structured data, a token count, or a model's context window, and never
learns whose model answered. There is no branch on provider name here and no
place to put one.

**This package depends on nothing else in the project.** Not the archive, not
storage, not the data model, not the importers. That is deliberate and it is
checked by a test: an abstraction that reached back into the layers meant to sit
above it would not be a boundary. Conversion between stored records and
``ChatMessage`` belongs to whichever later phase needs it.

**It also depends on no vendor SDK.** Providers register themselves by name, so
an implementation needing an optional dependency is available when installed and
simply absent when not. Only the deterministic fakes ship in the base package,
which is why the test suite needs no key, no network, and no local model server.

This phase establishes the interface and nothing that uses it. There is no
compaction here, no summarization, no extraction, and no context-pressure
detection; those are Phase 5 and later.
"""

from open_context.llm.config import (
    ENV_PREFIX,
    ProviderConfig,
    ProviderFactory,
    create_provider,
    from_env,
    provider_from_env,
    register_provider,
    registered_providers,
    unregister_provider,
)
from open_context.llm.errors import (
    AuthenticationError,
    ContextLimitExceededError,
    EmptyCompletionError,
    InvalidRequestError,
    LLMError,
    MalformedStructuredOutputError,
    ModelUnavailableError,
    ProviderUnavailableError,
    RateLimitError,
    TokenizationUnavailableError,
    UnsupportedCapabilityError,
)
from open_context.llm.info import Capability, ModelInfo
from open_context.llm.messages import ChatMessage, ChatRole
from open_context.llm.provider import (
    FinishReason,
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    StructuredResult,
    TokenUsage,
)
from open_context.llm.structured import parse_structured_output, schema_for
from open_context.llm.tokens import (
    CharacterRatioTokenizer,
    TokenCount,
    Tokenizer,
    UnavailableTokenizer,
    exact_counting_available,
)

__all__ = [
    "ENV_PREFIX",
    "AuthenticationError",
    "Capability",
    "CharacterRatioTokenizer",
    "ChatMessage",
    "ChatRole",
    "ContextLimitExceededError",
    "EmptyCompletionError",
    "FinishReason",
    "GenerationRequest",
    "GenerationResult",
    "InvalidRequestError",
    "LLMError",
    "LLMProvider",
    "MalformedStructuredOutputError",
    "ModelInfo",
    "ModelUnavailableError",
    "ProviderConfig",
    "ProviderFactory",
    "ProviderUnavailableError",
    "RateLimitError",
    "StructuredResult",
    "TokenCount",
    "TokenUsage",
    "TokenizationUnavailableError",
    "Tokenizer",
    "UnavailableTokenizer",
    "UnsupportedCapabilityError",
    "create_provider",
    "exact_counting_available",
    "from_env",
    "parse_structured_output",
    "provider_from_env",
    "register_provider",
    "registered_providers",
    "schema_for",
    "unregister_provider",
]
