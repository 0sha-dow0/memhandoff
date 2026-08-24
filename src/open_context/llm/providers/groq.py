"""The Groq provider.

Implements ``LLMProvider`` over Groq's chat-completions endpoint. Thin on
purpose: build a request, send it, translate the answer or the failure.

**No model is hard-coded.** Groq's available models change and differ by
account, so the model is whatever the caller configured, and the result records
the identifier Groq itself reported — which may differ from the one asked for.

**No context window is invented.** Groq does not return one with a completion,
so ``ModelInfo.context_window`` is whatever was configured explicitly and
``None`` otherwise. A plausible-looking guess would be a number every budget
decision downstream would trust.
"""

from __future__ import annotations

import json
import os
from typing import Any

from open_context.llm.config import ProviderConfig, register_provider
from open_context.llm.errors import (
    AuthenticationError,
    ProviderUnavailableError,
    UnsupportedCapabilityError,
)
from open_context.llm.free_models import BillingClass, require_free_model
from open_context.llm.info import Capability, ModelInfo
from open_context.llm.provider import GenerationRequest, GenerationResult, StructuredResult
from open_context.llm.providers import _chat
from open_context.llm.providers._http import (
    DEFAULT_TIMEOUT_SECONDS,
    USER_AGENT,
    HttpRequest,
    Transport,
    TransportError,
    urllib_transport,
)

PROVIDER_NAME = "groq"
API_KEY_ENV = "GROQ_API_KEY"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

CAPABILITIES = frozenset({Capability.TEXT_GENERATION})
"""What this provider will claim.

Text generation only. Structured output is not advertised because whether a
given Groq model honours a JSON mode varies by model and could not be verified
here, and a capability flag that is sometimes wrong is worse than one that is
absent. Exact token counting is absent because provider-reported usage arrives
after a call and is not a tokenizer.
"""


class GroqProvider:
    """Generates text through Groq."""

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        enforce_free: bool = True,
        transport: Transport = urllib_transport,
    ) -> None:
        # Before anything else, and before any credential is even looked at:
        # an unapproved model must cost zero network calls and zero money.
        self.billing_class = BillingClass.UNKNOWN
        if enforce_free:
            self.billing_class = require_free_model(PROVIDER_NAME, model).billing_class

        if not api_key:
            raise AuthenticationError(
                f"no Groq credential; set {API_KEY_ENV} or pass one in the configuration"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._api_key = api_key
        self._transport = transport
        self._info = ModelInfo(
            provider=PROVIDER_NAME,
            model=model,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            capabilities=CAPABILITIES,
        )

    def __repr__(self) -> str:
        """Never renders the credential."""
        return f"GroqProvider(model={self.model!r}, base_url={self.base_url!r})"

    # ------------------------------------------------------------------

    def model_info(self) -> ModelInfo:
        return self._info

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload = _chat.build_payload(request, self.model, provider=PROVIDER_NAME)
        response = self._post("/chat/completions", payload)
        _chat.raise_for_status(response, provider=PROVIDER_NAME, model=self.model)
        return _chat.parse_result(response, provider=PROVIDER_NAME, model=self.model)

    def structured_output(
        self,
        request: GenerationRequest,
        schema: Any,  # noqa: ANN401  JSON Schema
    ) -> StructuredResult:
        """Not implemented, deliberately.

        Groq exposes a JSON mode, but whether a given model honours a schema
        varies and none of it could be verified from here. Prompting for JSON
        and calling the result native structured output would make the
        capability flag a guess. The Phase 5.6 benchmark does not need it.
        """
        raise UnsupportedCapabilityError(
            Capability.STRUCTURED_OUTPUT.value, provider=PROVIDER_NAME, model=self.model
        )

    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict[str, Any]) -> Any:  # noqa: ANN401  HttpResponse
        request = HttpRequest(
            url=f"{self.base_url}{path}",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            body=json.dumps(payload).encode("utf-8"),
            timeout=self.timeout,
        )
        try:
            return self._transport(request)
        except TransportError as exc:
            raise ProviderUnavailableError(f"{PROVIDER_NAME}: {exc}") from exc


def build(config: ProviderConfig) -> GroqProvider:
    """Construct from configuration, taking the credential from the environment.

    An explicit ``api_key`` wins, then ``GROQ_API_KEY``. A missing credential is
    an error and never a quiet fall back to another provider or to the fake — a
    real-model run that silently became a deterministic one would produce
    numbers labelled as measurements.
    """
    options = dict(config.options)
    return GroqProvider(
        model=config.model,
        api_key=config.api_key or os.environ.get(API_KEY_ENV, ""),
        base_url=str(options.get("base_url") or config.base_url or DEFAULT_BASE_URL),
        context_window=_optional_int(options.get("context_window")),
        max_output_tokens=_optional_int(options.get("max_output_tokens")),
        timeout=float(options.get("timeout", DEFAULT_TIMEOUT_SECONDS)),
    )


def _optional_int(value: Any) -> int | None:  # noqa: ANN401  configuration values
    return int(value) if isinstance(value, int | str) and str(value).strip() else None


register_provider(PROVIDER_NAME, build, replace_existing=True)


__all__ = [
    "API_KEY_ENV",
    "CAPABILITIES",
    "DEFAULT_BASE_URL",
    "PROVIDER_NAME",
    "GroqProvider",
    "build",
]
