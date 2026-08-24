"""Provider selection and configuration.

No test here reads the real environment or needs a credential.
"""

import pytest

from open_context.llm import (
    ENV_PREFIX,
    ProviderConfig,
    ProviderUnavailableError,
    create_provider,
    from_env,
    provider_from_env,
    register_provider,
    registered_providers,
    unregister_provider,
)
from open_context.llm.fakes import PROVIDER_NAME, FakeProvider

ENV = {
    f"{ENV_PREFIX}PROVIDER": "fake",
    f"{ENV_PREFIX}MODEL": "fake-model",
}


@pytest.fixture
def temporary_provider():
    """Registers a name for one test and removes it afterwards."""
    registered: list[str] = []

    def register(name, factory):
        register_provider(name, factory)
        registered.append(name)

    yield register
    for name in registered:
        unregister_provider(name)


# ----------------------------------------------------------------------
# Configuration


def test_a_configuration_names_a_provider_and_a_model():
    config = ProviderConfig(provider="ollama", model="llama3")
    assert (config.provider, config.model) == ("ollama", "llama3")
    assert config.base_url is None
    assert not config.has_credentials


def test_a_configuration_must_name_both():
    with pytest.raises(ValueError, match="provider must be named"):
        ProviderConfig(provider="", model="m")
    with pytest.raises(ValueError, match="model must be named"):
        ProviderConfig(provider="p", model="  ")


def test_configuration_comes_from_the_environment():
    config = from_env({**ENV, f"{ENV_PREFIX}BASE_URL": "http://localhost:11434"})

    assert config.provider == "fake"
    assert config.model == "fake-model"
    assert config.base_url == "http://localhost:11434"


def test_a_missing_provider_is_an_explicit_failure():
    with pytest.raises(ProviderUnavailableError, match="no provider configured"):
        from_env({})


def test_a_missing_model_is_an_explicit_failure():
    with pytest.raises(ProviderUnavailableError, match="no model configured"):
        from_env({f"{ENV_PREFIX}PROVIDER": "fake"})


def test_a_default_provider_can_be_supplied():
    config = from_env({f"{ENV_PREFIX}MODEL": "m"}, default_provider="fake")
    assert config.provider == "fake"


def test_the_environment_wins_over_the_default():
    config = from_env(ENV, default_provider="something-else")
    assert config.provider == "fake"


def test_options_are_added_without_mutating():
    base = ProviderConfig(provider="fake", model="m")
    extended = base.with_options(reply="hi")

    assert extended.options == {"reply": "hi"}
    assert base.options == {}, "configurations are frozen"


# ----------------------------------------------------------------------
# Credentials


def test_an_api_key_is_kept_out_of_repr():
    """A key must not reach a log line, a traceback, or an assertion dump."""
    config = ProviderConfig(provider="p", model="m", api_key="sk-do-not-print-me")

    assert "sk-do-not-print-me" not in repr(config)
    assert config.has_credentials
    assert config.api_key == "sk-do-not-print-me", "still available to the provider"


def test_a_key_is_read_from_the_environment_never_hard_coded():
    config = from_env({**ENV, f"{ENV_PREFIX}API_KEY": "sk-from-env"})
    assert config.api_key == "sk-from-env"


def test_no_credential_is_needed_to_build_the_fake_provider():
    """The whole suite depends on this: offline, keyless, deterministic."""
    config = ProviderConfig(provider=PROVIDER_NAME, model="fake-model")
    assert not config.has_credentials

    provider = create_provider(config)
    assert provider.model_info().provider == "fake"


# ----------------------------------------------------------------------
# The registry


def test_the_fake_provider_registers_itself_on_import():
    assert PROVIDER_NAME in registered_providers()


def test_provider_selection_goes_through_the_registry():
    config = ProviderConfig(provider=PROVIDER_NAME, model="m-2").with_options(reply="scripted")
    provider = create_provider(config)

    assert provider.model_info().model == "m-2"
    assert provider.generate.__self__.reply == "scripted"  # type: ignore[attr-defined]


def test_an_unregistered_provider_names_what_is_available():
    """Usually a missing optional dependency, so the message has to be actionable."""
    with pytest.raises(ProviderUnavailableError) as info:
        create_provider(ProviderConfig(provider="not-installed", model="m"))

    message = str(info.value)
    assert "not-installed" in message
    assert PROVIDER_NAME in message
    assert "optional dependency" in message


def test_a_provider_can_be_registered_at_runtime(temporary_provider):
    """The seam an optional provider package uses: import, and it is available."""
    temporary_provider("in-test", lambda config: FakeProvider(model=config.model))

    assert "in-test" in registered_providers()
    assert create_provider(ProviderConfig(provider="in-test", model="m")).model_info().model == "m"


def test_registering_one_name_twice_is_refused(temporary_provider):
    """Otherwise which implementation wins would depend on import order."""
    temporary_provider("contested", lambda config: FakeProvider())

    with pytest.raises(ValueError, match="already registered"):
        register_provider("contested", lambda config: FakeProvider())


def test_an_empty_provider_name_is_refused():
    with pytest.raises(ValueError, match="must not be empty"):
        register_provider("  ", lambda config: FakeProvider())


def test_end_to_end_selection_from_the_environment():
    provider = provider_from_env(ENV)
    assert provider.model_info().model == "fake-model"


def test_registered_providers_is_sorted_and_stable():
    assert registered_providers() == tuple(sorted(registered_providers()))
