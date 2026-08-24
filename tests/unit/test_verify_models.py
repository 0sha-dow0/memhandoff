"""Re-checking the allowlist against a provider's catalogue.

Offline: the catalogue fetch is injected. This exists because the registry went
stale within ten days of being written — `openai/gpt-oss-20b:free` stopped being
free on OpenRouter — and a dated claim is not a checkable one.
"""

from open_context.llm.free_models import (
    BillingClass,
    Evidence,
    FreeModelSpec,
    Reasoning,
)
from open_context.llm.providers import verify as verify_module
from open_context.llm.providers.verify import Status, VerificationReport, verify


def spec(model_id="m:free", provider="openrouter", evidence=Evidence.ZERO_PRICE):
    return FreeModelSpec(
        provider=provider,
        model_id=model_id,
        billing_class=BillingClass.FREE,
        context_window=1000,
        verified_on="2026-08-23",
        verified_by="test",
        evidence=evidence,
        reasoning=Reasoning.NONE,
    )


def catalogue(monkeypatch, entries, *, fails=()):
    def fake(url, key):
        for provider, catalogue_url in verify_module.CATALOGUES.items():
            if catalogue_url == url:
                if provider in fails:
                    raise TimeoutError("no route to host")
                return {"data": entries.get(provider, [])}
        return {"data": []}

    monkeypatch.setattr(verify_module, "_fetch", fake)


def only(monkeypatch, specs):
    monkeypatch.setattr(verify_module, "APPROVED_FREE_MODELS", tuple(specs))


def test_a_zero_priced_model_is_confirmed(monkeypatch):
    only(monkeypatch, [spec()])
    catalogue(
        monkeypatch,
        {"openrouter": [{"id": "m:free", "pricing": {"prompt": "0", "completion": "0"}}]},
    )
    report = verify()
    assert report.results[0].status is Status.CONFIRMED
    assert report.sound


def test_a_model_that_started_charging_is_the_dangerous_case(monkeypatch):
    """This actually happened: `openai/gpt-oss-20b:free` stopped being free ten
    days after it was approved. An allowlist naming a priced model is the exact
    failure it exists to prevent."""
    only(monkeypatch, [spec()])
    catalogue(
        monkeypatch,
        {"openrouter": [{"id": "m:free", "pricing": {"prompt": "0.0000005", "completion": "0"}}]},
    )
    report = verify()
    assert report.results[0].status is Status.NOT_FREE
    assert not report.sound
    assert "AN APPROVED MODEL IS PRICED" in str(report)


def test_a_vanished_model_is_reported_but_does_not_fail_the_check(monkeypatch):
    """It cannot be called, so it cannot cost anything. Both Groq llama models
    disappeared this way — worth knowing, not a soundness failure."""
    only(monkeypatch, [spec()])
    catalogue(monkeypatch, {"openrouter": []})
    report = verify()
    assert report.results[0].status is Status.MISSING
    assert report.sound
    assert "gone from their catalogue" in str(report)


def test_an_account_tier_entry_is_reported_as_uncheckable(monkeypatch):
    """Groq prices every chat model it serves, so its free tier is an account
    allowance the API does not expose. Saying so beats a check that quietly
    passes on evidence it never had."""
    only(
        monkeypatch,
        [spec(model_id="openai/gpt-oss-20b", provider="groq", evidence=Evidence.ACCOUNT_TIER)],
    )
    catalogue(
        monkeypatch,
        {"groq": [{"id": "openai/gpt-oss-20b", "pricing": {"prompt": "0.0000001"}}]},
    )
    report = verify()
    assert report.results[0].status is Status.UNCHECKABLE
    assert report.sound, "a priced account-tier model is expected, not a failure"


def test_a_provider_that_cannot_be_reached_is_recorded_not_assumed(monkeypatch):
    """Silence is not confirmation. An unreachable catalogue must not read as
    'everything is fine'."""
    only(monkeypatch, [spec()])
    catalogue(monkeypatch, {}, fails={"openrouter", "groq"})
    report = verify()
    assert not report.results
    assert "openrouter" in report.unreachable
    assert "unreachable" in str(report)


def test_an_empty_report_is_sound_because_nothing_was_found_priced():
    assert VerificationReport().sound
