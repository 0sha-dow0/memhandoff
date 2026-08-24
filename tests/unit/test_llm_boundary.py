"""The architectural boundary, checked rather than agreed.

```
archive -> state -> future compaction -> LLMProvider / Tokenizer
```

and never

```
archive -> a vendor SDK
```

A dependency rule that lives only in a document is a rule that gets broken by
the first convenient import. These tests read the source and enforce it, so the
boundary fails at test time rather than at review time.
"""

import ast
from pathlib import Path

import pytest

import open_context

SOURCE_ROOT = Path(open_context.__file__).parent

LOWER_LAYERS = ("archive", "models", "storage", "importers")
"""Layers below compaction. None of them may reach a model.

``compaction`` is deliberately absent: it sits above the boundary and is
supposed to import ``open_context.llm``. What it may not do is import a vendor
SDK, which is checked separately.
"""

ABOVE_THE_BOUNDARY = ("compaction",)

VENDOR_SDKS = frozenset(
    {
        "openai",
        "anthropic",
        "ollama",
        "google",
        "cohere",
        "mistralai",
        "litellm",
        "langchain",
        "transformers",
        "tiktoken",
        "torch",
    }
)

NETWORK_MODULES = frozenset(
    {"socket", "ssl", "http", "urllib", "requests", "httpx", "aiohttp", "subprocess"}
)


def modules_in(package: str) -> list[Path]:
    return sorted((SOURCE_ROOT / package).rglob("*.py"))


def abstraction_modules() -> list[Path]:
    """The LLM abstraction itself, excluding provider implementations.

    ``llm/providers`` is where vendor code is *supposed* to live: a provider
    that talks to a real service necessarily opens a socket. The invariant that
    survives is narrower and still worth enforcing — the interface, the
    tokenizer contract, the errors, the configuration, and the fakes reach no
    network, so everything that only wants those pays nothing and the suite
    stays offline.
    """
    return [path for path in modules_in("llm") if "providers" not in path.parts]


def imports_of(path: Path) -> set[str]:
    """Every module name imported by a file, as written."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def top_level(module: str) -> str:
    return module.split(".", 1)[0]


def test_the_source_tree_is_actually_being_scanned():
    """A guard against these tests silently passing over nothing."""
    assert len(modules_in("llm")) >= 7
    for package in (*LOWER_LAYERS, *ABOVE_THE_BOUNDARY):
        assert modules_in(package), f"no modules found for {package}"


# ----------------------------------------------------------------------
# The core must not depend on any provider


@pytest.mark.parametrize("package", LOWER_LAYERS)
def test_the_core_does_not_import_the_llm_layer(package):
    """Nothing below compaction has any business calling a model.

    The archive stores what was said, storage indexes it, importers translate
    it. None of that needs a model, and a single import here would make the
    lower layers untestable without one.
    """
    for path in modules_in(package):
        offending = {name for name in imports_of(path) if name.startswith("open_context.llm")}
        assert not offending, f"{path.relative_to(SOURCE_ROOT)} imports {offending}"


@pytest.mark.parametrize("package", (*LOWER_LAYERS, *ABOVE_THE_BOUNDARY, "llm"))
def test_no_package_imports_a_vendor_sdk(package):
    """Not the core, not the abstraction, and not what sits above it.

    A provider implementation is the only place a vendor SDK may appear, and
    none ships in the base package. Compaction is the first consumer of the
    boundary and therefore the first place tempted to shortcut it.
    """
    for path in modules_in(package):
        offending = {name for name in imports_of(path) if top_level(name) in VENDOR_SDKS}
        assert not offending, f"{path.relative_to(SOURCE_ROOT)} imports {offending}"


@pytest.mark.parametrize("package", ABOVE_THE_BOUNDARY)
def test_a_consumer_reaches_models_only_through_the_boundary(package):
    """Compaction may call a model. It may not know which model it called.

    The rule the abstraction exists to make true: everything above the boundary
    depends on ``open_context.llm`` and on nothing behind it.
    """
    for path in modules_in(package):
        offending = {
            name
            for name in imports_of(path)
            if top_level(name) in VENDOR_SDKS or top_level(name) in NETWORK_MODULES
        }
        assert not offending, f"{path.relative_to(SOURCE_ROOT)} imports {offending}"


# ----------------------------------------------------------------------
# The abstraction must not depend on the core either


def test_the_llm_layer_does_not_import_the_rest_of_the_project():
    """It is a boundary, not a participant.

    The abstraction defines its own ``ChatMessage`` rather than reaching for
    ``models.Message``, so it can be lifted out, tested alone, and reused
    without dragging the archive along. Converting between the two belongs to
    whichever later phase needs it.
    """
    forbidden = tuple(f"open_context.{package}" for package in LOWER_LAYERS)
    for path in modules_in("llm"):
        offending = {name for name in imports_of(path) if name.startswith(forbidden)}
        assert not offending, f"{path.relative_to(SOURCE_ROOT)} imports {offending}"


def test_the_llm_abstraction_reaches_no_network_and_starts_no_process():
    """Why the suite needs no internet, no key, and no model server.

    The interface and the fakes are network-free, so importing them costs
    nothing and every test above the boundary runs offline. Provider
    implementations are excluded by construction: reaching a real service is
    what they are for, and they live in ``llm/providers`` precisely so this
    invariant can stay sharp for everything else.
    """
    for path in abstraction_modules():
        offending = {name for name in imports_of(path) if top_level(name) in NETWORK_MODULES}
        assert not offending, f"{path.relative_to(SOURCE_ROOT)} imports {offending}"


def test_the_abstraction_does_not_import_its_own_providers():
    """A caller that wants the interface must not get an HTTP client with it.

    ``open_context.llm`` importing ``open_context.llm.providers`` would drag
    network code into every consumer of the abstraction and quietly undo the
    test above.
    """
    for path in abstraction_modules():
        offending = {
            name for name in imports_of(path) if name.startswith("open_context.llm.providers")
        }
        assert not offending, f"{path.relative_to(SOURCE_ROOT)} imports {offending}"


def test_provider_implementations_exist_and_are_the_only_network_code():
    """A guard that the exclusion above is narrow and not accidentally total."""
    providers = modules_in("llm/providers")
    assert len(providers) >= 4, "the exclusion covers a real package, not an empty one"

    networked = [
        path
        for path in providers
        if {name for name in imports_of(path) if top_level(name) in NETWORK_MODULES}
    ]
    assert networked, "provider implementations are where network code is expected"


def test_the_llm_layer_depends_only_on_the_standard_library_and_pydantic():
    """Still true of everything, providers included. No SDK was added."""
    allowed = {"pydantic", "open_context", "__future__"}
    for path in modules_in("llm"):
        for name in imports_of(path):
            root = top_level(name)
            if root in allowed:
                continue
            assert root in _STDLIB, f"{path.relative_to(SOURCE_ROOT)} imports {name}"


_STDLIB = frozenset(
    {
        "abc",
        "ast",
        "collections",
        "contextlib",
        "dataclasses",
        "datetime",
        "enum",
        "functools",
        "hashlib",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "socket",
        "ssl",
        "time",
        "types",
        "typing",
        "urllib",
        "uuid",
    }
)


# ----------------------------------------------------------------------
# What the boundary buys


def test_code_can_be_written_against_the_interface_alone():
    """The property the whole phase exists for.

    A function that takes ``LLMProvider`` and ``Tokenizer`` works with the fake
    here and with a hosted model later, unchanged, and contains no branch on
    which. This is the shape a compaction engine will be written in; it is not
    a compaction engine, and nothing here summarizes, ranks, or selects.
    """
    from open_context.llm import ChatMessage, GenerationRequest, LLMProvider, Tokenizer
    from open_context.llm.fakes import FakeProvider, WordTokenizer

    def describe(provider: LLMProvider, tokenizer: Tokenizer, prompt: str) -> tuple[str, int, bool]:
        request = GenerationRequest.of(ChatMessage.user(prompt))
        counted = tokenizer.count_messages(request.messages)
        answer = provider.generate(request)
        return answer.text, counted.count, counted.exact

    text, count, exact = describe(FakeProvider(reply="ok"), WordTokenizer(), "one two three")

    assert text == "ok"
    assert count > 0
    assert exact
