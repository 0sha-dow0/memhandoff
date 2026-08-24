"""Phase 10. Compiling a `.ctx` package into context for a target agent.

    package = CtxPackage.from_json(path.read_text())
    context = compile_context(package, CompilationRequest(budget_tokens=2000, task=...), tokenizer)
    payload = render_for("anthropic", context)

Two layers, and the split between them is the design:

* **`portable`** decides. Selection and budgeting happen once, on a
  provider-neutral structure, using the same fitting logic as the compactor.
* **`targets`** renders. A renderer receives an already-fitted context with no
  budget and no tokenizer, so it can shape but not choose.

See ``docs/compiler.md``.
"""

from open_context.compiler.portable import (
    CompilationRequest,
    CompiledContext,
    CompiledSection,
    Section,
    compile_context,
)
from open_context.compiler.targets import (
    GENERIC,
    TARGETS,
    TargetRenderer,
    known_targets,
    render_anthropic,
    render_for,
    render_gemini,
    render_generic,
    render_local,
    render_openai,
)

__all__ = [
    "GENERIC",
    "TARGETS",
    "CompilationRequest",
    "CompiledContext",
    "CompiledSection",
    "Section",
    "TargetRenderer",
    "compile_context",
    "known_targets",
    "render_anthropic",
    "render_for",
    "render_gemini",
    "render_generic",
    "render_local",
    "render_openai",
]
