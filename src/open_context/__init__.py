"""Open Context Runtime.

A local-first context runtime for AI agents.

The runtime treats a model's context window as a working set rather than as
memory. Raw history lives in an archive, durable semantic state lives in a
state graph, and the context compiler assembles a task-specific context for
whichever model is being called.

Phase 0 contains no application logic. It exists to establish the package,
the test harness, and the licensing files.
"""

__version__ = "0.0.0"

__all__ = ["__version__"]
