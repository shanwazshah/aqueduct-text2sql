"""Run BIRD mini-dev through the crew.

Designed for the Kaggle tier: resumable, checkpointed after every question, and
tolerant of a session dying mid-sweep. A 12-hour session cap and a 30-hour weekly
GPU budget mean losing a run to an unhandled exception is expensive.

    python -m aqueduct.eval.bird_run --strategies direct --limit 150
    python -m aqueduct.eval.bird_run --strategies direct,chain,orchestrator --limit 150
    python -m aqueduct.eval.bird_run --report-only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

from ..crew import Crew, RepairMode
from ..strategies import STRATEGIES
from .bird import (
    BIRD_DIR,
    BirdQuestion,
    db_url_for,
    describe,
    find_databases,
    load_questions,
    schema_for,
    stratified_sample,
)
from .metrics import execution_accuracy

RESULTS_PATH = BIRD_DIR / "results.json"


@dataclass
class Row:
    strategy: str
    question_id: int
    db_id: str
    difficulty: str
    correct: bool
    draft_correct: bool
    reason: str
    sql: str
    calls: int
    seconds: float
    repaired: bool = False
    agents: list[str] = field(default_factory=list)

    # The draft query itself, not just whether it graded correct.
    #
    # `compare.py` has always stored this; this harness stored only the boolean,
    # which means a grader change cannot be applied to a finished sweep. The
    # gen EX column is the one every claim in the README rests on, so losing the
    # ability to re-derive it offline costs a full GPU re-run. Defaulted, so
    # results files written before this still load.
    draft_sql: str = ""


def load_rows(path: Path) -> dict[tuple[str, int], Row]:
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


def save_rows(rows: dict[tuple[str, int], Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps([asdict(r) for r in rows.values()], indent=1), encoding="utf-8")
    tmp.replace(path)  # atomic: a killed session cannot leave a half-written file


def run(
    questions: list[BirdQuestion],
    strategy_names: list[str],
    databases: dict[str, Path],
    *,
    repair: RepairMode = RepairMode.EXECUTION,
    path: Path = RESULTS_PATH,
) -> dict[tuple[str, int], Row]:
    rows = load_rows(path)

    for name in strategy_names:
        pending = [q for q in questions if (name, q.question_id) not in rows]
        print(f"\n{'=' * 70}\n  {name}   {len(pending)} of {len(questions)} remaining\n{'=' * 70}",
              flush=True)

        for i, question in enumerate(pending, 1):
            start = perf_counter()
            db_path = databases.get(question.db_id)

            if db_path is None:
                rows[(name, question.question_id)] = Row(
                    strategy=name, question_id=question.question_id, db_id=question.db_id,
                    difficulty=question.difficulty, correct=False, draft_correct=False,
                    reason=f"database '{question.db_id}' not found", sql="", calls=0,
                    seconds=0.0,
                )
                save_rows(rows, path)
                continue

            db_url = db_url_for(db_path)

            try:
                crew = Crew(
                    strategy=name,
                    repair=repair,
                    schema=schema_for(db_url),
                    db_url=db_url,
                    use_memory=False,
                )
                answer = crew.ask(question.prompt())
                grade = execution_accuracy(answer.sql, question.gold_sql, db_url=db_url)
                draft = answer.attempts[0].sql if answer.attempts else ""
                draft_grade = execution_accuracy(draft, question.gold_sql, db_url=db_url)

                row = Row(
                    strategy=name, question_id=question.question_id, db_id=question.db_id,
                    difficulty=question.difficulty, correct=grade.correct,
                    draft_correct=draft_grade.correct, reason=grade.reason,
                    sql=answer.sql, calls=answer.calls, seconds=perf_counter() - start,
                    repaired=answer.was_repaired, agents=answer.agents_used,
                    draft_sql=draft,
                )
            except Exception as e:
                # A 500-question sweep must not die on one bad question.
                row = Row(
                    strategy=name, question_id=question.question_id, db_id=question.db_id,
                    difficulty=question.difficulty, correct=False, draft_correct=False,
                    reason=f"crashed: {type(e).__name__}: {e}", sql="", calls=0,
                    seconds=perf_counter() - start,
                )

            rows[(name, question.question_id)] = row
            save_rows(rows, path)

            done = sum(1 for (s, _), r in rows.items() if s == name)
            correct = sum(1 for (s, _), r in rows.items() if s == name and r.correct)
            print(
                f"  [{i:>3}/{len(pending)}] q{question.question_id:<5} {question.difficulty:<12}"
                f"{'PASS' if row.correct else 'FAIL'}  {row.seconds:>5.1f}s   "
                f"running EX {100.0 * correct / done:5.1f}%   {row.reason[:38]}",
                flush=True,
            )

    return rows


def report(rows: dict[tuple[str, int], Row]) -> str:
    names = sorted({s for s, _ in rows})
    lines = [
        "",
        "=" * 88,
        "BIRD MINI-DEV".center(88),
        "=" * 88,
        f"{'strategy':<15}{'gen EX':>9}{'final EX':>10}{'rescued':>9}{'n':>6}"
        f"{'calls/q':>9}{'s/q':>8}{'simple':>9}{'moderate':>10}{'challenging':>12}",
        "-" * 88,
    ]

    for name in names:
        subset = [r for (s, _), r in rows.items() if s == name]
        n = len(subset)
        if not n:
            continue
        correct = sum(1 for r in subset if r.correct)
        gen = sum(1 for r in subset if r.draft_correct)

        by_difficulty = {}
        for level in ("simple", "moderate", "challenging"):
            group = [r for r in subset if r.difficulty == level]
            by_difficulty[level] = (
                f"{100.0 * sum(1 for r in group if r.correct) / len(group):.0f}%"
                if group else "-"
            )

        lines.append(
            f"{name:<15}{100.0 * gen / n:>8.1f}%{100.0 * correct / n:>9.1f}%"
            f"{sum(1 for r in subset if r.correct and not r.draft_correct):>9}{n:>6}"
            f"{sum(r.calls for r in subset) / n:>9.1f}"
            f"{sum(r.seconds for r in subset) / n:>8.1f}"
            f"{by_difficulty['simple']:>9}{by_difficulty['moderate']:>10}"
            f"{by_difficulty['challenging']:>12}"
        )

    lines.append("=" * 88)

    crashes = [r for r in rows.values() if r.reason.startswith("crashed")]
    if crashes:
        lines.append(f"\ncrashes: {len(crashes)}")
        for reason, count in Counter(r.reason[:70] for r in crashes).most_common(5):
            lines.append(f"  {count:>3}x {reason}")

    missing_db = [r for r in rows.values() if "not found" in r.reason]
    if missing_db:
        lines.append(f"\nmissing databases: {len(missing_db)}")
        for db, count in Counter(r.db_id for r in missing_db).most_common():
            lines.append(f"  {count:>3}x {db}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BIRD mini-dev.")
    parser.add_argument("--strategies", default="direct")
    parser.add_argument("--limit", type=int, default=150, help="0 for all 500.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repair", default=RepairMode.EXECUTION.value)
    parser.add_argument("--databases", default=str(BIRD_DIR))
    parser.add_argument("--questions", default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        print(report(load_rows(RESULTS_PATH)))
        return

    names = [n.strip() for n in args.strategies.split(",") if n.strip()]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise SystemExit(f"unknown strategies: {', '.join(unknown)}")

    questions = load_questions(Path(args.questions) if args.questions else None)
    if args.limit:
        questions = stratified_sample(questions, args.limit, seed=args.seed)

    databases = find_databases(Path(args.databases))
    print(f"questions: {describe(questions)}")
    print(f"databases found: {len(databases)}")

    needed = {q.db_id for q in questions}
    missing = sorted(needed - set(databases))
    if missing:
        print(f"WARNING missing {len(missing)} databases: {', '.join(missing)}")

    rows = run(questions, names, databases, repair=RepairMode(args.repair))
    print(report(rows))


if __name__ == "__main__":
    main()
