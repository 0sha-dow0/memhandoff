"""The OpenRouter provider.

Implements ``LLMProvider`` over OpenRouter's chat-completions endpoint.

**This is an OpenRouter provider, not a generic OpenAI-compatible one.** The
wire dialect happens to be shared with Groq, and the translation for it lives in
a private helper both call. What does not exist is a configurable
`OpenAICompatibleProvider` that OpenRouter is an instance of: this class owns its
name, its credential, its endpoint, its optional attribution headers, and its
own decisions, and the runtime depends on none of OpenAI's conventions.

**Any model identifier is accepted.** OpenRouter's catalogue and its free tier
change without notice, so nothing here names a model. The caller configures one
and the result records the identifier OpenRouter reported back.

**No context window is invented.** OpenRouter publishes context lengths through
a separate models endpoint that this provider deliberately does not call — one
request per generation, no hidden second call. Configure it explicitly, or leave
it unknown.
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

PROVIDER_NAME = "openrouter"
API_KEY_ENV = "OPENROUTER_API_KEY"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

REFERER_ENV = "OPENROUTER_REFERRER"
TITLE_ENV = "OPENROUTER_TITLE"
"""Optional attribution OpenRouter accepts. Absent unless configured."""

CAPABILITIES = frozenset({Capability.TEXT_GENERATION})
"""Text generation only.

Structured output is not advertised: OpenRouter routes to many models whose
support for a JSON schema differs, so a flag set here would be right for some
routes and wrong for others. Exact token counting is absent because
provider-reported usage is not a tokenizer.
"""


class OpenRouterProvider:
    """Generates text through OpenRouter."""

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
        referrer: str | None = None,
        title: str | None = None,
        transport: Transport = urllib_transport,
    ) -> None:
        # Before anything else, and before any credential is even looked at:
        # an unapproved model must cost zero network calls and zero money.
        self.billing_class = BillingClass.UNKNOWN
        if enforce_free:
            self.billing_class = require_free_model(PROVIDER_NAME, model).billing_class

        if not api_key:
            raise AuthenticationError(
                f"no OpenRouter credential; set {API_KEY_ENV} or pass one in the configuration"
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.referrer = referrer
        self.title = title
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
        return f"OpenRouterProvider(model={self.model!r}, base_url={self.base_url!r})"

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

        OpenRouter forwards to many models with different structured-output
        support, so honouring the contract for one route and not another would
        make the capability a lie half the time. Prompting for JSON and calling
        it native would be worse. The Phase 5.6 benchmark does not need it.
        """
        raise UnsupportedCapabilityError(
            Capability.STRUCTURED_OUTPUT.value, provider=PROVIDER_NAME, model=self.model
        )

    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        }
        if self.referrer:
            headers["HTTP-Referer"] = self.referrer
        if self.title:
            headers["X-Title"] = self.title
        return headers

    def _post(self, path: str, payload: dict[str, Any]) -> Any:  # noqa: ANN401  HttpResponse
        request = HttpRequest(
            url=f"{self.base_url}{path}",
            headers=self._headers(),
            body=json.dumps(payload).encode("utf-8"),
            timeout=self.timeout,
        )
        try:
            return self._transport(request)
        except TransportError as exc:
            raise ProviderUnavailableError(f"{PROVIDER_NAME}: {exc}") from exc


def build(config: ProviderConfig) -> OpenRouterProvider:
    """Construct from configuration, taking the credential from the environment.

    An explicit ``api_key`` wins, then ``OPENROUTER_API_KEY``. A missing
    credential is an error and never a quiet fall back to another provider or to
    the fake.
    """
    options = dict(config.options)
    return OpenRouterProvider(
        model=config.model,
        api_key=config.api_key or os.environ.get(API_KEY_ENV, ""),
        base_url=str(options.get("base_url") or config.base_url or DEFAULT_BASE_URL),
        context_window=_optional_int(options.get("context_window")),
        max_output_tokens=_optional_int(options.get("max_output_tokens")),
        timeout=float(options.get("timeout", DEFAULT_TIMEOUT_SECONDS)),
        referrer=options.get("referrer") or os.environ.get(REFERER_ENV) or None,
        title=options.get("title") or os.environ.get(TITLE_ENV) or None,
    )


def _optional_int(value: Any) -> int | None:  # noqa: ANN401  configuration values
    return int(value) if isinstance(value, int | str) and str(value).strip() else None


register_provider(PROVIDER_NAME, build, replace_existing=True)


__all__ = [
    "API_KEY_ENV",
    "CAPABILITIES",
    "DEFAULT_BASE_URL",
    "PROVIDER_NAME",
    "OpenRouterProvider",
    "build",
]
