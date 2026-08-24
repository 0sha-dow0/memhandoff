"""Structured output, normalized.

Providers reach structured output by different routes: a native structured mode,
a JSON mode, grammar-constrained decoding, or nothing at all and a prompt that
asks politely. The route is the provider's business. What crosses the boundary
is the same either way — parsed data, or ``MalformedStructuredOutputError``.

**JSON Schema is the neutral schema language.** Not because it is pleasant, but
because every provider that supports structured output at all accepts it, and
because pydantic emits it, so a caller with a model already has one. The core
never sees a provider's schema dialect.

**What this module validates, exactly.** ``parse_structured_output`` parses JSON
and checks the shape the schema declares at its top level: the ``type``, and
the presence of ``required`` properties. That is a guard against the common
failures — prose instead of JSON, a fenced code block, an object missing half
its fields — and it is *not* full JSON Schema validation. Nested constraints,
formats, enums, and value types are not checked. A caller needing real
validation validates the returned data itself, which with pydantic is one
``model_validate`` call. Saying so here is better than a helper that implies a
guarantee it does not provide.

Nothing in this module knows what is being extracted. There are no prompts here
and no domain schemas; building those is a later phase's work.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from open_context.llm.errors import MalformedStructuredOutputError

TEXT_SAMPLE_LENGTH = 500
"""How much of an unusable response an error carries."""

_JSON_TYPES: Mapping[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
}


def schema_for(model: type[Any]) -> dict[str, Any]:
    """JSON Schema for a pydantic model.

    A convenience so a caller can pass a model rather than hand-writing a
    schema. The LLM layer still speaks only JSON Schema, so this takes the model
    and returns the neutral form immediately; nothing downstream learns that
    pydantic was involved.
    """
    generator = getattr(model, "model_json_schema", None)
    if generator is None:
        raise TypeError(f"{model!r} is not a pydantic model")
    schema: dict[str, Any] = generator()
    return schema


def _strip_code_fence(text: str) -> str:
    """Remove a surrounding markdown fence, which models add unprompted."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    newline = body.find("\n")
    if newline == -1:
        return stripped
    body = body[newline + 1 :]
    closing = body.rfind("```")
    return body[:closing].strip() if closing != -1 else body.strip()


def parse_structured_output(text: str, schema: Mapping[str, Any]) -> Any:  # noqa: ANN401
    """Parse a model's response into data, or say why it could not.

    Shape checking only; see the module docstring for exactly how far it goes.
    """
    candidate = _strip_code_fence(text)
    if not candidate:
        raise MalformedStructuredOutputError("the model returned no content", text=text)

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise MalformedStructuredOutputError(
            f"response is not valid JSON: {exc}", text=text[:TEXT_SAMPLE_LENGTH]
        ) from exc

    _check_shape(data, schema, text)
    return data


def _check_shape(data: Any, schema: Mapping[str, Any], text: str) -> None:  # noqa: ANN401
    declared = schema.get("type")
    if isinstance(declared, str):
        expected = _JSON_TYPES.get(declared)
        # bool is a subclass of int, so an integer schema must not accept True.
        wrong_type = expected is not None and not isinstance(data, expected)
        if wrong_type or (declared in {"number", "integer"} and isinstance(data, bool)):
            raise MalformedStructuredOutputError(
                f"response should be a JSON {declared}, got {type(data).__name__}",
                text=text[:TEXT_SAMPLE_LENGTH],
            )

    required = schema.get("required")
    if isinstance(data, dict) and isinstance(required, list):
        missing = [key for key in required if key not in data]
        if missing:
            raise MalformedStructuredOutputError(
                f"response is missing required {'field' if len(missing) == 1 else 'fields'} "
                f"{', '.join(str(key) for key in sorted(missing))}",
                text=text[:TEXT_SAMPLE_LENGTH],
            )


__all__ = ["TEXT_SAMPLE_LENGTH", "parse_structured_output", "schema_for"]
