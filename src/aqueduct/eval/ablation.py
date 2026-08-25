"""Repair-mode ablation.

The question this answers: **which feedback signal actually repairs SQL?**

The source notebooks assume the answer is model self-critique — `critique_sql`
produces a JSON verdict and `refine_sql` acts on it. Phase 1 gave a reason to
doubt that at 3B scale, so this measures the four modes against each other
instead of picking one:

    NONE       write once, run once (the Phase 1 control)
    EXECUTION  repair only on database errors
    CRITIQUE   repair only on the Critic's opinion (what the notebooks do)
    BOTH       execution first, then critique

Controls that make the comparison meaningful:

  * Every mode gets its own empty memory, so a lesson learned in one run cannot
    leak into another and inflate it.
  * Temperature is 0 and responses are cached, so the *first* draft is identical
    across all four modes. Any difference in score is therefore attributable to
    repair, not to sampling luck.
  * Cost is reported alongside accuracy. A mode that gains two points for triple
    the calls has not obviously won, and reporting only accuracy would hide that.

    python -m aqueduct.eval.ablation
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..agents.memory import ErrorMemory
from ..crew import Crew, RepairMode
from ..db.introspect import load_schema
from .demo_set import ALL_QUESTIONS
from .metrics import Report
from .runner import evaluate, failures


@dataclass
class ModeResult:
    mode: RepairMode
    report: Report
    repaired: int
    total_attempts: int
    failure_detail: str

    @property
    def calls_per_question(self) -> float:
        return self.report.llm_calls / self.report.total if self.report.total else 0.0


def run_mode(mode: RepairMode, schema, *, verbose: bool = False) -> ModeResult:
    """Evaluate one repair mode against the demo set with a clean memory."""
    memory = ErrorMemory(Path(tempfile.mkdtemp(prefix="aq-mem-")) / "memory.json")
    crew = Crew(repair=mode, schema=schema, memory=memory)

    report, rows = evaluate(lambda q: crew.ask(q), ALL_QUESTIONS, verbose=verbose)

    return ModeResult(
        mode=mode,
        report=report,
        repaired=sum(1 for _, a, _ in rows if a.was_repaired),
        total_attempts=sum(a.attempt_count for _, a, _ in rows),
        failure_detail=failures(rows),
    )


def main() -> None:
    schema = load_schema()  # shared, so introspection cost is not counted per mode
    results: list[ModeResult] = []

    for mode in (RepairMode.NONE, RepairMode.EXECUTION, RepairMode.CRITIQUE, RepairMode.BOTH):
        print(f"\n{'=' * 64}\n  repair mode: {mode.value}\n{'=' * 64}")
        result = run_mode(mode, schema, verbose=True)
        results.append(result)
        print(f"\n  EX {result.report.ex:.1f}%  ({result.report.correct}/{result.report.total})"
              f"   repaired {result.repaired}"
              f"   calls/q {result.calls_per_question:.1f}"
              f"   {result.report.seconds:.0f}s")

    print(f"\n\n{'=' * 76}")
    print("  REPAIR MODE COMPARISON".center(76))
    print("=" * 76)
    header = f"{'mode':<12}{'EX':>8}{'correct':>10}{'repaired':>10}{'calls/q':>10}{'time':>10}"
    print(header)
    print("-" * 76)

    baseline = results[0].report.ex
    for r in results:
        delta = r.report.ex - baseline
        delta_text = "  --  " if r.mode is RepairMode.NONE else f"{delta:+.1f}"
        print(
            f"{r.mode.value:<12}"
            f"{r.report.ex:>7.1f}%"
            f"{r.report.correct:>7}/{r.report.total:<2}"
            f"{r.repaired:>10}"
            f"{r.calls_per_question:>10.1f}"
            f"{r.report.seconds:>9.0f}s"
            f"   {delta_text}"
        )
    print("=" * 76)

    for r in results:
        if r.report.correct < r.report.total:
            print(f"\n--- failures under {r.mode.value} ---")
            print(r.failure_detail)


if __name__ == "__main__":
    main()
