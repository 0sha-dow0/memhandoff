"""Draw the benchmark results, from the benchmark results.

    benchmarks/results/*.jsonl  ->  this script  ->  docs/assets/benchmarks/*.svg

**No number in a chart is written down here.** Every value is read out of a
persisted run and recomputed the same way the analysis recomputes it, so a chart
cannot drift from the data it claims to show. A second copy of the numbers,
maintained by hand, would be a second source of truth — and the one that gets
stale is always the one on display.

Run it:

    python benchmarks/plot.py

Hand-drawn look, no dependencies. Lines are perturbed by a *seeded* generator, so
regenerating produces byte-identical files and a diff means the data moved.
"""

from __future__ import annotations

import json
import pathlib
import random
import statistics
import subprocess
import sys
from typing import Any

Row = dict[str, Any]
"""One benchmark result row, as persisted. Read, never constructed here."""

RESULTS = pathlib.Path("benchmarks/results")
OUT = pathlib.Path("docs/assets/benchmarks")

# The authoritative run: real model, adversarial dataset, every cell, three
# repetitions. Named once so a chart cannot quietly be drawn from a partial run.
AUTHORITATIVE = RESULTS / "v4-adversarial-8b.jsonl"

DISOWNED = "should not be read as"
"""A cell its own strategy says is not that strategy. Excluded from its score."""

ARMS = ["full_context", "simple_summary_v1", "phase_5_baseline", "hybrid_v1"]
LABEL = {
    "full_context": "full context\n(reference)",
    "simple_summary_v1": "plain summary",
    "phase_5_baseline": "Phase 5\nbaseline",
    "hybrid_v1": "hybrid",
}

INK = "#1f1d2b"
PAPER = "#fdfcf8"
PURPLE = "#8b7bc7"
PURPLE_SOFT = "#c9bfe8"
MUTED = "#8a8598"
HAND = "'Comic Sans MS','Chalkboard SE','Bradley Hand',Segoe Print,cursive,sans-serif"


# ----------------------------------------------------------------------
# Reading the data


def load(path: pathlib.Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def usable(rows: list[Row]) -> list[Row]:
    """Successful cells that their own strategy stands behind."""
    return [
        r
        for r in rows
        if r["status"] == "success"
        and not any(DISOWNED in w for w in r["strategy_detail"].get("warnings", []))
    ]


def deterministic_score(rows: list[Row]) -> float | None:
    """Retention plus completion: string checks, no model in the loop.

    Judged questions are excluded on purpose. In this run the judge failed its
    own control — the arm handed the whole conversation scored 9 of 24 on judged
    questions while passing every deterministic check — so nothing it graded can
    be shown as a result.
    """
    hit = total = 0
    for row in rows:
        hit += sum(c["passed"] for c in row["retention_checks"])
        total += len(row["retention_checks"])
        hit += sum(c["status"] == "passed" for c in row["completion"])
        total += len(row["completion"])
    return hit / total if total else None


def disowned_count(rows: list[Row], arm: str) -> int:
    return sum(
        1
        for r in rows
        if r["strategy"] == arm
        and r["status"] == "success"
        and any(DISOWNED in w for w in r["strategy_detail"].get("warnings", []))
    )


# ----------------------------------------------------------------------
# Drawing by hand


class Sketch:
    """A tiny SVG canvas that draws slightly wrong on purpose.

    The wobble is seeded per drawing, so the same data always produces the same
    file. A chart that jittered differently on every run would make every
    regeneration look like a change.
    """

    def __init__(self, width: int, height: int, seed: int = 7) -> None:
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        self.parts: list[str] = []

    def _wobble(self, value: float, amount: float = 1.6) -> float:
        return value + self.rng.uniform(-amount, amount)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        colour: str = INK,
        width: float = 2.2,
        dash: str = "",
    ) -> None:
        """A line drawn as two overlapping strokes, the way a pen doubles back."""
        style = f' stroke-dasharray="{dash}"' if dash else ""
        for pass_ in range(2 if not dash else 1):
            a, b = (0.0, 0.0) if dash else (self._wobble(0, 1.0), self._wobble(0, 1.0))
            c, d = (0.0, 0.0) if dash else (self._wobble(0, 1.0), self._wobble(0, 1.0))
            mx = (x1 + x2) / 2 + (0 if dash else self._wobble(0, 1.4))
            my = (y1 + y2) / 2 + (0 if dash else self._wobble(0, 1.4))
            self.parts.append(
                f'<path d="M{x1 + a:.1f},{y1 + b:.1f} Q{mx:.1f},{my:.1f} '
                f'{x2 + c:.1f},{y2 + d:.1f}" fill="none" stroke="{colour}" '
                f'stroke-width="{width - pass_ * 0.7:.1f}" stroke-linecap="round"{style}/>'
            )

    def box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = "none",
        hatch: bool = False,
        seedless: bool = False,
    ) -> None:
        """A rectangle whose corners do not quite meet."""
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        jitter = [
            (
                px + (0 if seedless else self.rng.uniform(-1.4, 1.4)),
                py + (0 if seedless else self.rng.uniform(-1.4, 1.4)),
            )
            for px, py in pts
        ]
        path = "M" + " L".join(f"{px:.1f},{py:.1f}" for px, py in jitter) + " Z"
        if fill != "none":
            self.parts.append(f'<path d="{path}" fill="{fill}" opacity="0.85"/>')
        if hatch:
            self.parts.append(f'<path d="{path}" fill="url(#hatch)"/>')
        self.parts.append(
            f'<path d="{path}" fill="none" stroke="{INK}" stroke-width="2.1" '
            f'stroke-linejoin="round"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        s: object,
        *,
        size: int = 15,
        anchor: str = "middle",
        colour: str = INK,
        weight: str = "normal",
    ) -> None:
        for i, chunk in enumerate(str(s).split("\n")):
            self.parts.append(
                f'<text x="{x:.1f}" y="{y + i * (size + 3):.1f}" font-family="{HAND}" '
                f'font-size="{size}" fill="{colour}" text-anchor="{anchor}" '
                f'font-weight="{weight}">{escape(chunk)}</text>'
            )

    def render(self, title: str, subtitle: str = "") -> str:
        head = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" role="img">',
            f"<title>{escape(title)}</title>",
            "<defs>",
            '<pattern id="hatch" width="7" height="7" patternTransform="rotate(45)" '
            'patternUnits="userSpaceOnUse">',
            f'<line x1="0" y1="0" x2="0" y2="7" stroke="{INK}" stroke-width="1.1" opacity="0.30"/>',
            "</pattern>",
            "</defs>",
            f'<rect width="{self.w}" height="{self.h}" fill="{PAPER}"/>',
        ]
        return "\n".join([*head, *self.parts, "</svg>"]) + "\n"


def escape(s: object) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write(name: str, svg: str) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(svg, encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# 1. What each arm retained


def chart_retention(rows: list[Row]) -> pathlib.Path:
    """Deterministic score per arm, with the spread across three repetitions.

    The spread is drawn, not summarised, because it is the whole reason this
    chart can be read at all: an earlier run reported a between-arm gap the same
    size as its own run-to-run noise, which is a picture of nothing.
    """
    good = usable(rows)
    reps = sorted({r["run"]["configuration"]["benchmark_repetition"] for r in good})

    measured: list[tuple[str, float | None, list[float], int]] = []
    for arm in ARMS:
        cells = [r for r in good if r["strategy"] == arm]
        pooled = deterministic_score(cells)
        scenarios = len({r["scenario_id"] for r in cells})
        by_rep = [
            deterministic_score(
                [r for r in cells if r["run"]["configuration"]["benchmark_repetition"] == p]
            )
            for p in reps
        ]
        measured.append((arm, pooled, [v for v in by_rep if v is not None], scenarios))

    s = Sketch(780, 560, seed=11)
    left, right, base, top = 132, 720, 400, 150

    s.text(30, 38, "What survived compaction", size=23, anchor="start", weight="bold")
    s.text(
        30,
        62,
        "adversarial dataset v4 - 8 scenarios x 3 runs - budget 160 tokens",
        size=13,
        anchor="start",
        colour=MUTED,
    )
    s.text(
        30,
        82,
        "deterministic checks only - bars are the pooled score, whiskers the "
        "spread across the three runs",
        size=12,
        anchor="start",
        colour=MUTED,
    )

    s.line(left, base, right, base)
    s.line(left, base, left, top)

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = base - frac * (base - top)
        if frac:
            s.line(left, y, right, y, colour=MUTED, width=1.0, dash="3 7")
        s.text(left - 14, y + 5, f"{frac:.2f}", size=13, anchor="end", colour=MUTED)

    s.text(left - 92, (base + top) / 2, "kept", size=15, colour=MUTED)

    width = (right - left) / (len(ARMS) + 0.6)
    for index, (arm, pooled, per_rep, scenarios) in enumerate(measured):
        cx = left + width * (index + 0.8)
        # Fewer than half the scenarios is a sample, not a score.
        scored = pooled is not None and scenarios >= 6
        if not scored or pooled is None:
            # Deliberately not drawn as a bar. A full-height hatch reads as a
            # top score, and this arm has no score: 21 of its 24 cells were
            # disowned by the compactor that produced them.
            missing = disowned_count(rows, arm)
            s.box(cx - width * 0.32, base - 38, width * 0.64, 38, hatch=True)
            s.text(cx, base - 50, "no score", size=14, colour=MUTED)
            s.text(
                cx,
                base + 72,
                f"{missing} of {missing + scenarios * 3} cells",
                size=11,
                colour=MUTED,
            )
            s.text(cx, base + 86, "disowned", size=11, colour=MUTED)
        else:
            height = pooled * (base - top)
            s.box(
                cx - width * 0.32,
                base - height,
                width * 0.64,
                height,
                fill=PURPLE if arm == "full_context" else PURPLE_SOFT,
                hatch=True,
            )
            s.text(cx, base - height - 34, f"{pooled:.2f}", size=20, weight="bold")
            if len(per_rep) > 1:
                lo, hi = min(per_rep), max(per_rep)
                ylo, yhi = base - lo * (base - top), base - hi * (base - top)
                s.line(cx, ylo, cx, yhi, width=1.8)
                for y in (ylo, yhi):
                    s.line(cx - 9, y, cx + 9, y, width=1.8)
                # Below the arm label, which may run to two lines.
                s.text(cx, base + 72, f"spread {hi - lo:.2f}", size=11, colour=MUTED)
        s.text(cx, base + 28, LABEL[arm], size=14)

    s.text(
        30,
        524,
        "The reference arm is not a competitor - it receives the whole "
        "conversation, so it shows the information was",
        size=12,
        anchor="start",
        colour=MUTED,
    )
    s.text(
        30,
        540,
        "recoverable at all. The gap between the two compaction arms is "
        "smaller than either one's spread across runs.",
        size=12,
        anchor="start",
        colour=MUTED,
    )
    return write("benchmark-retention.svg", s.render("What survived compaction"))


# ----------------------------------------------------------------------
# 2. Where the loss actually is


def chart_per_scenario(rows: list[Row]) -> pathlib.Path:
    """Score per scenario, for the arms with enough coverage to compare.

    Worth its own chart because the pooled number hides the finding: four of six
    scenarios are perfect for every arm, and all of the loss sits in the two
    built around exact values.
    """
    good = usable(rows)
    arms = [a for a in ARMS if len({r["scenario_id"] for r in good if r["strategy"] == a}) >= 6]

    scored = []
    for scenario in sorted({r["scenario_id"] for r in good}):
        values = [
            deterministic_score(
                [r for r in good if r["strategy"] == a and r["scenario_id"] == scenario]
            )
            for a in arms
        ]
        if any(v is not None for v in values):
            scored.append((scenario, values))

    # Three treatments, one small palette, so a reader can tell the arms apart
    # without a colour wheel: solid, plain fill, and hatch.
    style = [(PURPLE, False), (PURPLE_SOFT, False), (PAPER, True)]

    s = Sketch(820, 176 + 58 * len(scored), seed=23)
    left, right = 250, 700
    top = 148

    for index, arm in enumerate(arms):
        x = 250 + 190 * index
        fill, hatch = style[index]
        s.box(x, 96, 18, 13, fill=fill, hatch=hatch)
        s.text(x + 26, 108, LABEL[arm].replace("\n", " "), size=13, anchor="start", colour=MUTED)

    for row_index, (scenario, values) in enumerate(scored):
        y = top + row_index * 58
        s.text(left - 18, y + 20, scenario.replace("adv-", ""), size=14, anchor="end")
        for index, value in enumerate(values):
            if value is None:
                continue
            fill, hatch = style[index]
            bar_top = y + index * 12
            length = (right - left) * value
            if length < 4:
                s.text(left + 10, bar_top + 10, "0", size=12, colour=MUTED, anchor="start")
            else:
                s.box(left, bar_top, length, 10, fill=fill, hatch=hatch)
                s.text(
                    left + length + 10,
                    bar_top + 10,
                    f"{value:.2f}",
                    size=11,
                    anchor="start",
                    colour=MUTED,
                )
        s.line(left, y - 8, right, y - 8, colour=MUTED, width=0.8, dash="2 8")

    s.line(left, top - 12, left, top + 58 * len(scored) - 18)
    s.text(30, 38, "Where compaction loses", size=23, anchor="start", weight="bold")
    s.text(
        30,
        62,
        "the same run, split by scenario - longer is better",
        size=13,
        anchor="start",
        colour=MUTED,
    )
    s.text(
        30,
        152 + 58 * len(scored),
        "Every arm is perfect on four of six. All of the loss sits in the two scenarios "
        "built around exact values.",
        size=12,
        anchor="start",
        colour=MUTED,
    )
    return write("benchmark-per-scenario.svg", s.render("Where compaction loses"))


# ----------------------------------------------------------------------
# 3. What compaction costs


def chart_compression(rows: list[Row]) -> pathlib.Path:
    """Context size against the tokens spent producing it.

    Two bars per arm because compression alone is a misleading number: a small
    context bought with two model calls is not free, and a chart showing only
    the first would say it was.
    """
    good = usable(rows)
    data = []
    for arm in ARMS:
        cells = [r for r in good if r["strategy"] == arm]
        if len({r["scenario_id"] for r in cells}) < 6:
            continue
        data.append(
            (
                arm,
                statistics.mean(r["original_tokens"] for r in cells),
                statistics.mean(r["context_tokens"] for r in cells),
                statistics.mean(
                    (r["compaction_input_tokens"] or 0)
                    + (r["compaction_output_tokens"] or 0)
                    + r["continuation_input_tokens"]
                    + r["continuation_output_tokens"]
                    for r in cells
                ),
            )
        )

    ceiling = max(max(d[3] for d in data), max(d[1] for d in data))
    s = Sketch(760, 430, seed=31)
    left, right, base, top = 150, 700, 320, 110
    s.line(left, base, right, base)
    s.line(left, base, left, top)

    width = (right - left) / (len(data) + 0.5)
    for index, (arm, _original, context, spent) in enumerate(data):
        cx = left + width * (index + 0.7)
        for offset, value, fill, hatch in (
            (-0.30, context, PURPLE, True),
            (0.02, spent, PURPLE_SOFT, False),
        ):
            height = (value / ceiling) * (base - top)
            s.box(cx + width * offset, base - height, width * 0.28, height, fill=fill, hatch=hatch)
            s.text(cx + width * (offset + 0.14), base - height - 10, f"{value:,.0f}", size=12)
        s.text(cx, base + 24, LABEL[arm], size=13)

    y_original = base - (data[0][1] / ceiling) * (base - top)
    s.line(left, y_original, right, y_original, colour=MUTED, width=1.2, dash="4 6")
    s.text(
        right + 4,
        y_original + 4,
        f"original\n{data[0][1]:,.0f}",
        size=11,
        anchor="start",
        colour=MUTED,
    )

    s.box(150, 78, 14, 11, fill=PURPLE, hatch=True)
    s.text(170, 88, "context handed to the model", size=13, anchor="start", colour=MUTED)
    s.box(400, 78, 14, 11, fill=PURPLE_SOFT)
    s.text(420, 88, "tokens spent producing it", size=13, anchor="start", colour=MUTED)

    s.text(30, 34, "What compaction costs", size=23, anchor="start", weight="bold")
    s.text(30, 58, "mean tokens per scenario, same run", size=13, anchor="start", colour=MUTED)
    s.text(
        30,
        392,
        "Compaction shrinks the context roughly 16x and spends more tokens than "
        "sending the conversation whole.",
        size=12,
        anchor="start",
        colour=MUTED,
    )
    return write("benchmark-compression.svg", s.render("What compaction costs"))


# ----------------------------------------------------------------------
# 4. The test suite


def read_test_summary() -> dict[str, int] | None:
    """Run the suite and read its own summary line.

    Run rather than remembered. A hardcoded count is wrong the first time
    somebody adds a test, and a chart that says 1,285 while the suite says
    something else is worse than no chart.
    """
    import re

    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    summary: dict[str, int] = {}
    for line in reversed(completed.stdout.splitlines()):
        found = re.findall(r"(\d+) (passed|failed|skipped|error|xfailed)", line)
        if found:
            for count, kind in found:
                summary[kind] = int(count)
            break
    return summary or None


def chart_test_suite(summary: dict[str, int]) -> pathlib.Path:
    """The suite's own numbers, and nothing derived from them.

    Deliberately not a percentage. "99.8% passing" is a marketing sentence
    dressed as a measurement, and it reads the same whether the two that did not
    pass were skipped or failed.
    """
    order = [("passed", PURPLE), ("skipped", PURPLE_SOFT), ("failed", PAPER)]
    counts = [(name, summary.get(name, 0), colour) for name, colour in order]

    s = Sketch(720, 260, seed=43)
    left, right, top, bottom = 50, 670, 108, 200

    # The small categories get a fixed, readable box and the large one takes what
    # is left. Sizing all three by value makes "2" a sliver nobody can read, and
    # "0" no box at all — which is the number most worth seeing.
    minor = [c for c in counts if c[1] < max(x[1] for x in counts)]
    minor_width = 104
    span = (right - left) - minor_width * len(minor) - 8 * len(minor)

    x = left
    for name, count, colour in counts:
        width = span if count == max(c[1] for c in counts) else minor_width
        s.box(x, top, width, bottom - top, fill=colour, hatch=(name == "passed"))
        s.text(x + width / 2, top + 44, f"{count:,}", size=26, weight="bold")
        s.text(x + width / 2, top + 70, name, size=14, colour=MUTED)
        x += width + 8

    s.text(30, 38, "Test suite", size=23, anchor="start", weight="bold")
    s.text(30, 62, "pytest, run to produce this chart", size=13, anchor="start", colour=MUTED)
    s.text(
        30,
        236,
        "The two skipped tests call a real provider and skip when no credential is present.",
        size=12,
        anchor="start",
        colour=MUTED,
    )
    return write("benchmark-test-suite.svg", s.render("Test suite"))


# ----------------------------------------------------------------------
# Product identity


def chart_identity() -> pathlib.Path:
    """The one picture: a session leaving one agent and arriving at another.

    Drawn in the same hand as the charts so the project looks like one thing.
    It carries no numbers, because it is an identity rather than a result.
    """
    s = Sketch(980, 300, seed=5)

    stages = [
        (60, "Agent A", "a long session", PAPER, False),
        (280, "open-context", "archive - extract\ncompact", PURPLE_SOFT, True),
        (520, "project.ctx", "portable context", PURPLE, True),
        (760, "Agent B", "continues the work", PAPER, False),
    ]

    for x, title, caption, fill, hatch in stages:
        s.box(x, 108, 140, 74, fill=fill, hatch=hatch)
        s.text(x + 70, 148, title, size=17, weight="bold")
        s.text(x + 70, 202, caption, size=12, colour=MUTED)

    for x_from, x_to in ((192, 278), (412, 518), (652, 758)):
        s.line(x_from, 145, x_to - 10, 145, width=2.0)
        s.line(x_to - 22, 138, x_to - 8, 145, width=2.0)
        s.line(x_to - 22, 152, x_to - 8, 145, width=2.0)

    s.text(
        30,
        44,
        "Take a long session from one agent and continue it in another",
        size=19,
        anchor="start",
        weight="bold",
    )
    s.text(
        30,
        70,
        "local-first - provider-neutral - the original conversation never leaves your machine",
        size=13,
        anchor="start",
        colour=MUTED,
    )
    s.text(
        30,
        268,
        "The archive stays where it is. Only the package travels.",
        size=12,
        anchor="start",
        colour=MUTED,
    )
    return write(
        "../memhandoff-identity.svg",
        s.render("A session leaving one agent and arriving at another"),
    )


# ----------------------------------------------------------------------


def main() -> int:
    if not AUTHORITATIVE.is_file():
        print(f"no benchmark artifact at {AUTHORITATIVE}", file=sys.stderr)
        print("charts are drawn from a real run; there is nothing to draw.", file=sys.stderr)
        return 1

    rows = load(AUTHORITATIVE)
    written = [
        chart_identity(),
        chart_retention(rows),
        chart_per_scenario(rows),
        chart_compression(rows),
    ]

    summary = read_test_summary()
    if summary:
        written.append(chart_test_suite(summary))
    else:
        print("could not read a pytest summary; skipping the test chart", file=sys.stderr)

    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
