"""Real provider implementations.

```
import open_context.llm.providers   # registers groq and openrouter
```

Importing this package registers every provider in it. Nothing above the
boundary knows they exist: code asks for a provider by name and gets one, and
there is no `if provider == "groq"` anywhere outside these modules.

**Deliberately not imported by ``open_context.llm``.** The abstraction stays
free of network code, so anything that only wants the interface, the tokenizer
contract, or the deterministic fakes pays nothing for the existence of an HTTP
client. A caller that wants to reach a real model imports this package on
purpose.

**No new dependency.** Both providers post JSON over ``urllib``. An SDK per
vendor would be a larger surface for a smaller benefit, and the base package
keeping its single runtime dependency is worth more.

Credentials come from the environment, never from code. See docs/llm.md.
"""

from open_context.llm.providers import groq, openrouter
from open_context.llm.providers._http import (
    DEFAULT_TIMEOUT_SECONDS,
    HttpRequest,
    HttpResponse,
    Transport,
    TransportError,
    urllib_transport,
)
from open_context.llm.providers.groq import GroqProvider
from open_context.llm.providers.openrouter import OpenRouterProvider

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "GroqProvider",
    "HttpRequest",
    "HttpResponse",
    "OpenRouterProvider",
    "Transport",
    "TransportError",
    "groq",
    "openrouter",
    "urllib_transport",
]
