"""Which importers are tried, and in what order.

**Order is the whole content of this module.** ``JsonLinesImporter.detect``
accepts any line that parses as a JSON object, which every Claude Code
transcript also does — so a generic-first registry would claim every session
file and import it as anonymous records, losing tool calls, the parent chain,
and the branch structure, while appearing to succeed.

Specific formats therefore come first and the generic reader is the fallback it
was always meant to be. It lives here rather than in ``generic`` because a
registry naming both would make that module import ``claude_code``, which
already imports it.
"""

from __future__ import annotations

from typing import Final

from open_context.importers.base import ConversationImporter
from open_context.importers.claude_code import ClaudeCodeImporter
from open_context.importers.generic import JsonLinesImporter

DEFAULT_IMPORTERS: Final[tuple[ConversationImporter, ...]] = (
    ClaudeCodeImporter(),
    JsonLinesImporter(),
)

__all__ = ["DEFAULT_IMPORTERS"]
