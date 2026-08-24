"""Choosing a provider.

```
ProviderConfig -> registry -> LLMProvider
```

A name and a model, resolved through a registry to an implementation. That
indirection is what keeps vendor SDKs out of the base package: a provider
registers itself when its module is imported, and a module nobody imports costs
nothing. Installing an extra makes a provider available; not installing it makes
that provider's name unknown, which is an ordinary, explainable failure rather
than an ``ImportError`` from the middle of a call stack.

**Configuration stays small on purpose.** Environment variables and a frozen
dataclass. No config server, no secrets manager, no database, no file format to
version. A local-first tool that needed infrastructure to decide which model to
call would have lost the plot.

**Credentials are never written down here.** ``api_key`` is read from the
environment and is excluded from ``repr``, so it cannot leak into a log line, a
traceback, or a pytest assertion dump by accident. Nothing in this repository
contains a key, and nothing should: the field exists to carry one from the
environment to a provider and no further.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from open_context.llm.errors import ProviderUnavailableError
from open_context.llm.provider import LLMProvider

ENV_PREFIX = "OPEN_CONTEXT_"
"""Namespace for every environment variable this project reads."""

ProviderFactory = Callable[["ProviderConfig"], LLMProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


@dataclass(frozen=True)
class ProviderConfig:
    """Which provider and model to use, and how to reach them."""

    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("provider must be named")
        if not self.model.strip():
            raise ValueError("model must be named")

    def with_options(self, **options: Any) -> ProviderConfig:  # noqa: ANN401
        return replace(self, options={**self.options, **options})

    @property
    def has_credentials(self) -> bool:
        """Whether a key was supplied. Never reveals the key itself."""
        return bool(self.api_key)


def from_env(
    environ: Mapping[str, str] | None = None, *, default_provider: str | None = None
) -> ProviderConfig:
    """Build a configuration from the environment.

    Reads ``OPEN_CONTEXT_PROVIDER``, ``OPEN_CONTEXT_MODEL``,
    ``OPEN_CONTEXT_BASE_URL``, and ``OPEN_CONTEXT_API_KEY``. ``environ`` is
    injectable so tests never touch the real environment and never need a key.
    """
    source = os.environ if environ is None else environ
    provider = source.get(f"{ENV_PREFIX}PROVIDER") or default_provider
    if not provider:
        raise ProviderUnavailableError(
            f"no provider configured; set {ENV_PREFIX}PROVIDER or pass one explicitly"
        )
    model = source.get(f"{ENV_PREFIX}MODEL")
    if not model:
        raise ProviderUnavailableError(f"no model configured; set {ENV_PREFIX}MODEL")
    return ProviderConfig(
        provider=provider,
        model=model,
        base_url=source.get(f"{ENV_PREFIX}BASE_URL"),
        api_key=source.get(f"{ENV_PREFIX}API_KEY"),
    )


def register_provider(
    name: str, factory: ProviderFactory, *, replace_existing: bool = False
) -> None:
    """Make a provider available under a name.

    A provider module calls this on import. Re-registering a name is refused
    unless asked for, so two implementations cannot silently claim one name and
    leave which-one-wins depending on import order.
    """
    if not name.strip():
        raise ValueError("a provider name must not be empty")
    if name in _REGISTRY and not replace_existing:
        raise ValueError(f"provider {name!r} is already registered")
    _REGISTRY[name] = factory


def unregister_provider(name: str) -> None:
    """Remove a provider. For tests, and for nothing else."""
    _REGISTRY.pop(name, None)


def registered_providers() -> tuple[str, ...]:
    """Provider names available in this process, sorted."""
    return tuple(sorted(_REGISTRY))


def create_provider(config: ProviderConfig) -> LLMProvider:
    """Build the provider named by a configuration.

    An unregistered name is ``ProviderUnavailableError``, listing what is
    available. That is usually a missing optional dependency, so the message
    has to be enough to work that out.
    """
    factory = _REGISTRY.get(config.provider)
    if factory is None:
        available = ", ".join(registered_providers()) or "none"
        raise ProviderUnavailableError(
            f"no provider registered as {config.provider!r}; available: {available}. "
            "A provider needing an optional dependency registers itself when installed."
        )
    return factory(config)


def provider_from_env(
    environ: Mapping[str, str] | None = None, *, default_provider: str | None = None
) -> LLMProvider:
    """Configuration and construction in one step, for the common case."""
    return create_provider(from_env(environ, default_provider=default_provider))


def _iter_registry() -> Iterator[tuple[str, ProviderFactory]]:
    yield from sorted(_REGISTRY.items())


__all__ = [
    "ENV_PREFIX",
    "ProviderConfig",
    "ProviderFactory",
    "create_provider",
    "from_env",
    "provider_from_env",
    "register_provider",
    "registered_providers",
    "unregister_provider",
]
