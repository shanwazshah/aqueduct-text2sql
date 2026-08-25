"""Strategy comparison — the leaderboard.

Runs every generation strategy over the same questions with the same grader and
the same repair setting, and reports accuracy *next to* cost. Reporting accuracy
alone would hide the entire point: `orchestrator` spends seven calls where
`direct` spends one, and whether that is worth it is the question Phase 4's
router exists to answer.

Controls:

  * Repair is held constant (`EXECUTION` by default), so a difference between two
    strategies is attributable to generation rather than to how they recover.
  * Memory is off. Phase 2 showed retrieval affecting unrelated questions; with
    it on, results would depend on question order.
  * Temperature 0 and a response cache, so a re-run reproduces exactly.

Results are checkpointed after every question. A sweep is a couple of hours on
the dev tier and the session can be interrupted; resuming re-reads what is
already done rather than paying for it twice.

    python -m aqueduct.eval.compare
    python -m aqueduct.eval.compare --strategies direct,chain
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from ..config import DATA_DIR
from ..crew import Crew, RepairMode
from ..db.introspect import load_schema
from ..strategies import STRATEGIES
from .demo_set import ALL_QUESTIONS, BY_ID
from .metrics import execution_accuracy

RESULTS_PATH = DATA_DIR / "comparison.json"


@dataclass
class Row:
    """One strategy's attempt at one question."""

    strategy: str
    question_id: str
    correct: bool
    reason: str
    sql: str
    calls: int
    seconds: float
    agents: list[str] = field(default_factory=list)
    repaired: bool = False


def load_rows(path: Path) -> dict[tuple[str, str], Row]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    rows = {}
    for entry in raw:
        row = Row(**entry)
        rows[(row.strategy, row.question_id)] = row
    return rows


def save_rows(rows: dict[tuple[str, str], Row], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(r) for r in rows.values()], indent=2), encoding="utf-8"
    )


def run(
    strategy_names: list[str],
    *,
    repair: RepairMode = RepairMode.EXECUTION,
    path: Path = RESULTS_PATH,
    resume: bool = True,
) -> dict[tuple[str, str], Row]:
    schema = load_schema()
    rows = load_rows(path) if resume else {}

    for name in strategy_names:
        pending = [q for q in ALL_QUESTIONS if (name, q.id) not in rows]
        if not pending:
            print(f"{name:<14} already complete ({len(ALL_QUESTIONS)} questions)")
            continue

        print(f"\n{'=' * 66}\n  {name}  ({len(pending)} to run)\n{'=' * 66}")

        for i, question in enumerate(pending, 1):
            crew = Crew(strategy=name, repair=repair, schema=schema, use_memory=False)
            start = perf_counter()
            try:
                answer = crew.ask(question.question)
                grade = execution_accuracy(answer.sql, question.gold_sql)
                row = Row(
                    strategy=name,
                    question_id=question.id,
                    correct=grade.correct,
                    reason=grade.reason,
                    sql=answer.sql,
                    calls=answer.usage.calls,
                    seconds=perf_counter() - start,
                    agents=answer.agents_used,
                    repaired=answer.was_repaired,
                )
            except Exception as e:
                # One strategy blowing up on one question must not lose the
                # whole sweep. Record it as a failure and carry on.
                row = Row(
                    strategy=name,
                    question_id=question.id,
                    correct=False,
                    reason=f"crashed: {type(e).__name__}: {e}",
                    sql="",
                    calls=0,
                    seconds=perf_counter() - start,
                )

            rows[(name, question.id)] = row
            save_rows(rows, path)  # checkpoint every question

            mark = "PASS" if row.correct else "FAIL"
            print(
                f"  [{i:>2}/{len(pending)}] {question.id}  {mark}  "
                f"{row.calls:>2} calls  {row.seconds:>5.1f}s  {row.reason[:44]}"
            )

    return rows


def report(rows: dict[tuple[str, str], Row], strategy_names: list[str]) -> str:
    lines = [
        "",
        "=" * 82,
        "STRATEGY COMPARISON".center(82),
        "=" * 82,
        f"{'strategy':<15}{'EX':>8}{'correct':>10}{'calls/q':>10}{'s/q':>9}"
        f"{'repairs':>9}{'vs direct':>12}",
        "-" * 82,
    ]

    baseline = None
    for name in strategy_names:
        subset = [r for (s, _), r in rows.items() if s == name]
        if not subset:
            continue
        total = len(subset)
        correct = sum(1 for r in subset if r.correct)
        ex = 100.0 * correct / total
        if name == "direct":
            baseline = ex

        delta = "  --  " if baseline is None or name == "direct" else f"{ex - baseline:+.1f}"
        lines.append(
            f"{name:<15}{ex:>7.1f}%{correct:>7}/{total:<2}"
            f"{sum(r.calls for r in subset) / total:>10.1f}"
            f"{sum(r.seconds for r in subset) / total:>9.1f}"
            f"{sum(1 for r in subset if r.repaired):>9}"
            f"{delta:>12}"
        )

    lines.append("=" * 82)

    # Which questions nothing solved: those describe the real ceiling.
    by_question: dict[str, list[Row]] = {}
    for (_, qid), row in rows.items():
        by_question.setdefault(qid, []).append(row)

    never = sorted(q for q, rs in by_question.items() if not any(r.correct for r in rs))
    always = sorted(q for q, rs in by_question.items() if all(r.correct for r in rs))

    lines += [
        "",
        f"solved by every strategy ({len(always)}): {', '.join(always) or 'none'}",
        f"solved by none ({len(never)}): {', '.join(never) or 'none'}",
    ]

    contested = sorted(set(by_question) - set(always) - set(never))
    if contested:
        lines += ["", "contested questions - where strategy choice actually matters:"]
        for qid in contested:
            winners = sorted(r.strategy for r in by_question[qid] if r.correct)
            question = BY_ID[qid]
            lines.append(f"  {qid} [{question.trap or question.difficulty}]: {', '.join(winners)}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generation strategies.")
    parser.add_argument("--strategies", default=",".join(STRATEGIES))
    parser.add_argument("--repair", default=RepairMode.EXECUTION.value)
    parser.add_argument("--fresh", action="store_true", help="Ignore saved results.")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")

    if args.report_only:
        rows = load_rows(RESULTS_PATH)
    else:
        rows = run(names, repair=RepairMode(args.repair), resume=not args.fresh)

    print(report(rows, names))


if __name__ == "__main__":
    main()
