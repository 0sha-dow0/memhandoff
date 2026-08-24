"""A very small synchronous HTTP client.

``urllib`` from the standard library. No new dependency, because a provider that
posts one JSON document and reads one back does not need an SDK, and the base
package staying at a single runtime dependency is worth more than the ergonomics
of one that does.

**Transport is injectable.** A provider takes a callable rather than reaching for
``urllib`` directly, so its tests exercise the real translation and error mapping
against a fake transport instead of monkeypatching the standard library. Nothing
in the test suite opens a socket.

**No retries, no pooling, no background anything.** One request, one response, a
timeout. Retry policy is deliberately absent from the provider boundary; see
docs/llm.md.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

DEFAULT_TIMEOUT_SECONDS = 60.0

USER_AGENT = "open-context-runtime"
"""How this client identifies itself.

Not decoration. Providers sit behind edge protection that rejects a request with
no ``User-Agent`` — Groq answers a bare ``urllib`` request with a Cloudflare 403
before it ever reaches the API — and identifying the client is ordinary HTTP
manners besides. Sent by both providers rather than injected by the transport,
so a test sees the same headers the service will.
"""


@dataclass(frozen=True)
class HttpRequest:
    url: str
    headers: Mapping[str, str]
    body: bytes
    timeout: float = DEFAULT_TIMEOUT_SECONDS

    def redacted_headers(self) -> dict[str, str]:
        """Headers with any credential replaced.

        The only form in which headers may be logged, put in an error, or shown
        to anyone. A traceback that helpfully printed the request would
        otherwise print the key.
        """
        return {
            name: ("<redacted>" if name.lower() in {"authorization", "x-api-key"} else value)
            for name, value in self.headers.items()
        }


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


Transport = Callable[[HttpRequest], HttpResponse]


class TransportError(Exception):
    """The request never produced an HTTP response.

    A timeout, a refused connection, DNS failure. Distinct from a response with
    an error status, which is a thing the provider said rather than a failure to
    reach it, and is translated separately.
    """


def urllib_transport(request: HttpRequest) -> HttpResponse:
    """Send one request. The default transport.

    An error status arrives as a response rather than an exception, so status
    translation lives in one place per provider instead of being split between
    a success path and an exception handler.
    """
    prepared = urllib.request.Request(
        request.url, data=request.body, headers=dict(request.headers), method="POST"
    )
    try:
        with urllib.request.urlopen(prepared, timeout=request.timeout) as response:
            return HttpResponse(
                status=int(response.status),
                body=response.read(),
                headers={key.lower(): value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        return HttpResponse(
            status=int(exc.code),
            body=exc.read(),
            headers={key.lower(): value for key, value in exc.headers.items()}
            if exc.headers
            else {},
        )
    except urllib.error.URLError as exc:
        raise TransportError(f"could not reach {request.url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise TransportError(f"{request.url} timed out after {request.timeout}s") from exc


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "HttpRequest",
    "HttpResponse",
    "Transport",
    "TransportError",
    "urllib_transport",
]
