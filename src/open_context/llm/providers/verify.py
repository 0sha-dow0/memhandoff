"""Re-checking the free-model registry against the live catalogues.

**A dated claim is not a checkable one.** The registry records when each entry
was verified and by what, which is better than nothing and still asks a reader to
trust a date. This turns that into something they can run.

It is worth running. Between one verification and the next, ten days apart,
`openai/gpt-oss-20b:free` stopped being free on OpenRouter and both non-reasoning
Groq models disappeared from the account — so the registry was, briefly, naming a
model that would have cost money. That is exactly the failure the allowlist
exists to prevent, and it was found by looking rather than by remembering.

**Reads catalogues, never generates.** The only requests made are the providers'
own model listings, which are free on both. Nothing here calls a model, so
verifying costs nothing but a listing.

**It lives under ``providers`` because it reaches the network.** The abstraction
above this directory is network-free by construction — that is why the whole test
suite runs offline without a key — and a verifier written into ``llm/`` broke
that invariant the moment it imported ``urllib``. The boundary test caught it.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from enum import StrEnum

from open_context.llm.free_models import APPROVED_FREE_MODELS, Evidence, FreeModelSpec
from open_context.llm.providers._http import USER_AGENT

CATALOGUES = {
    "openrouter": "https://openrouter.ai/api/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
}


class Status(StrEnum):
    CONFIRMED = "confirmed"
    """Present, and its price is still exactly zero."""

    NOT_FREE = "not_free"
    """Present and now priced. **The dangerous one.**"""

    MISSING = "missing"
    """Gone from the catalogue. Calls would fail rather than cost money."""

    UNCHECKABLE = "uncheckable"
    """Present, but the catalogue cannot settle the question.

    Groq prices every chat model it serves, so its free tier is an account-level
    allowance the API does not expose. An `ACCOUNT_TIER` entry can be confirmed
    to *exist* and never to be free, and saying so is more useful than a check
    that quietly passes.
    """


@dataclass(frozen=True)
class Result:
    spec: FreeModelSpec
    status: Status
    detail: str = ""

    def __str__(self) -> str:
        return f"[{self.status}] {self.spec.provider}/{self.spec.model_id}: {self.detail}"


@dataclass
class VerificationReport:
    results: list[Result] = field(default_factory=list)
    unreachable: dict[str, str] = field(default_factory=dict)

    @property
    def not_free(self) -> list[Result]:
        return [r for r in self.results if r.status is Status.NOT_FREE]

    @property
    def missing(self) -> list[Result]:
        return [r for r in self.results if r.status is Status.MISSING]

    @property
    def sound(self) -> bool:
        """No approved entry is now priced.

        A missing model is not a soundness failure — it cannot be called, so it
        cannot cost anything. A priced one can.
        """
        return not self.not_free

    def __str__(self) -> str:
        lines = [str(result) for result in self.results]
        for provider, reason in sorted(self.unreachable.items()):
            lines.append(f"[unreachable] {provider}: {reason}")
        verdict = "no approved model is priced" if self.sound else "AN APPROVED MODEL IS PRICED"
        lines.append(f"\n{verdict}")
        if self.missing:
            lines.append(
                f"{len(self.missing)} approved models are gone from their catalogue; "
                f"calls would fail rather than bill"
            )
        return "\n".join(lines)


def _fetch(url: str, key: str | None) -> dict[str, object]:
    headers = {"User-Agent": USER_AGENT}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=45) as response:
        document = json.load(response)
    return document if isinstance(document, dict) else {}


def _priced(entry: dict[str, object]) -> bool | None:
    """Whether a catalogue entry charges. ``None`` when it does not say."""
    pricing = entry.get("pricing")
    if not isinstance(pricing, dict):
        return None
    try:
        prompt = float(pricing.get("prompt", 0))
        completion = float(pricing.get("completion", 0))
    except (TypeError, ValueError):
        return None
    return prompt > 0 or completion > 0


def verify(keys: dict[str, str | None] | None = None) -> VerificationReport:
    """Check every approved entry against its provider's catalogue."""
    credentials = keys or {}
    report = VerificationReport()

    catalogues: dict[str, dict[str, dict[str, object]]] = {}
    for provider, url in CATALOGUES.items():
        try:
            document = _fetch(url, credentials.get(provider))
        except Exception as exc:
            report.unreachable[provider] = f"{type(exc).__name__}: {exc}"
            continue
        data = document.get("data")
        entries = data if isinstance(data, list) else []
        catalogues[provider] = {
            str(e.get("id")): e for e in entries if isinstance(e, dict) and e.get("id")
        }

    for spec in APPROVED_FREE_MODELS:
        catalogue = catalogues.get(spec.provider)
        if catalogue is None:
            continue
        entry = catalogue.get(spec.model_id)
        if entry is None:
            report.results.append(Result(spec, Status.MISSING, "not in the catalogue any more"))
            continue

        priced = _priced(entry)
        if spec.evidence is Evidence.ACCOUNT_TIER:
            report.results.append(
                Result(
                    spec,
                    Status.UNCHECKABLE,
                    "exists; freeness is an account allowance the API does not expose",
                )
            )
        elif priced:
            report.results.append(
                Result(spec, Status.NOT_FREE, "the catalogue now reports a non-zero price")
            )
        else:
            report.results.append(Result(spec, Status.CONFIRMED, "price is still zero"))

    return report


__all__ = ["CATALOGUES", "Result", "Status", "VerificationReport", "verify"]
