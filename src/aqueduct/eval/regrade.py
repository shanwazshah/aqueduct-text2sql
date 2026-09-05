"""Re-grade a finished sweep, without spending a single LLM call.

A results file stores the SQL each strategy produced. That means a change to the
grader can be applied to work already done: re-run the stored queries against the
databases, compare to the reference again, and report what moved. No GPU, no
model, no waiting.

This exists because the Phase 7 audit found the grader relaxing row order
whenever the *prediction* omitted `ORDER BY`, which let a prediction opt out of
the check. Fixing that makes the grader stricter, so every number measured before
the fix is due a re-grade. Re-running the sweeps to find out would cost hours of
GPU for a question that is pure arithmetic over data already on disk.

**What can and cannot be re-graded.** Both the final query and the pre-repair
draft are needed to reproduce the leaderboard's two columns:

    final EX   from `sql`        - stored by both harnesses, always re-gradable
    gen EX     from `draft_sql`  - stored by `compare.py` from the start, and by
                                   `bird_run.py` only from Phase 7 onward

A sweep run before that change cannot have its gen EX re-derived, because the
draft was never written down. Those rows are reported as *not re-gradable*
rather than silently carried over as unchanged - a re-grade that quietly reuses
an ungradable number is exactly the kind of instrument bug this project keeps
finding.

    python -m aqueduct.eval.regrade --results data/comparison.json
    python -m aqueduct.eval.regrade --results data/bird/bird_results_7b.json --databases data/bird
    python -m aqueduct.eval.regrade --results data/comparison.json --write
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from .metrics import execution_accuracy


@dataclass
class Change:
    """One row whose grade moved."""

    strategy: str
    question_id: object
    column: str          # "final" or "gen"
    was: bool
    now: bool

    @property
    def direction(self) -> str:
        return "pass->FAIL" if self.was else "FAIL->pass"


@dataclass
class Tally:
    """Re-grade outcome for one strategy."""

    strategy: str
    rows: int = 0
    final_was: int = 0
    final_now: int = 0
    gen_was: int = 0
    gen_now: int = 0
    gen_ungradable: int = 0
    skipped: int = 0
    changes: list[Change] = field(default_factory=list)

    def pct(self, n: int) -> float:
        return 100.0 * n / self.rows if self.rows else 0.0


def _gold_for_bird(questions_path: Path | None) -> dict:
    from .bird import load_questions

    return {q.question_id: q.gold_sql for q in load_questions(questions_path)}


def _gold_for_demo() -> dict:
    from .demo_set import BY_ID

    return {qid: q.gold_sql for qid, q in BY_ID.items()}


def regrade(
    results_path: Path,
    *,
    databases_root: Path | None = None,
    questions_path: Path | None = None,
) -> tuple[dict[str, Tally], list[dict]]:
    """Re-grade every row. Returns tallies per strategy and the updated rows.

    The results file is not modified; the caller decides whether to write.
    """
    raw = json.loads(results_path.read_text(encoding="utf-8"))
    if not raw:
        raise SystemExit(f"{results_path} holds no rows.")

    # The two harnesses write different shapes. BIRD rows carry the database
    # they were answered against; demo-set rows all share one.
    is_bird = "db_id" in raw[0]
    db_url_for = None

    if is_bird:
        if databases_root is None:
            raise SystemExit(
                "BIRD results need --databases pointing at the directory holding "
                "the .sqlite files (they are not in the repository)."
            )
        from .bird import db_url_for, find_databases

        gold = _gold_for_bird(questions_path)
        databases = find_databases(databases_root)
        if not databases:
            raise SystemExit(f"no .sqlite databases found under {databases_root}")
    else:
        gold = _gold_for_demo()
        databases = {}

    tallies: dict[str, Tally] = {}
    updated: list[dict] = []

    for entry in raw:
        row = dict(entry)
        updated.append(row)

        tally = tallies.setdefault(row["strategy"], Tally(strategy=row["strategy"]))

        gold_sql = gold.get(row["question_id"])
        db_url = None
        if is_bird:
            db_path = databases.get(row["db_id"])
            db_url = db_url_for(db_path) if db_path else None
            if db_path is None:
                gold_sql = None

        if not gold_sql:
            # No reference, or no database to run it against. Not a flip - just
            # a row this re-grade cannot speak to.
            tally.skipped += 1
            continue

        tally.rows += 1

        was_final = bool(row.get("correct"))
        now_final = execution_accuracy(row.get("sql") or "", gold_sql, db_url=db_url).correct
        tally.final_was += was_final
        tally.final_now += now_final
        if was_final != now_final:
            tally.changes.append(
                Change(row["strategy"], row["question_id"], "final", was_final, now_final)
            )
        row["correct"] = now_final

        draft_sql = row.get("draft_sql")
        if draft_sql is None:
            # Pre-Phase-7 bird_run rows. The stored boolean is left alone and
            # reported separately; it is not evidence under the current grader.
            tally.gen_ungradable += 1
            continue

        was_gen = bool(row.get("draft_correct"))
        now_gen = execution_accuracy(draft_sql, gold_sql, db_url=db_url).correct
        tally.gen_was += was_gen
        tally.gen_now += now_gen
        if was_gen != now_gen:
            tally.changes.append(
                Change(row["strategy"], row["question_id"], "gen", was_gen, now_gen)
            )
        row["draft_correct"] = now_gen

    return tallies, updated


def report(tallies: dict[str, Tally]) -> str:
    lines = [
        "",
        "=" * 78,
        "RE-GRADE".center(78),
        "=" * 78,
        f"{'strategy':<15}{'n':>5}{'gen was':>10}{'gen now':>10}"
        f"{'final was':>12}{'final now':>11}{'moved':>8}",
        "-" * 78,
    ]

    for name in sorted(tallies):
        t = tallies[name]
        gradable = t.rows - t.gen_ungradable
        gen_was = f"{100.0 * t.gen_was / gradable:.1f}%" if gradable else "-"
        gen_now = f"{100.0 * t.gen_now / gradable:.1f}%" if gradable else "-"
        lines.append(
            f"{name:<15}{t.rows:>5}{gen_was:>10}{gen_now:>10}"
            f"{t.pct(t.final_was):>11.1f}%{t.pct(t.final_now):>10.1f}%"
            f"{len(t.changes):>8}"
        )

    lines.append("=" * 78)

    ungradable = sum(t.gen_ungradable for t in tallies.values())
    if ungradable:
        lines += [
            "",
            f"gen EX not re-gradable for {ungradable} rows - written before the "
            "draft query",
            "was stored, so their gen EX is not evidence under the current grader.",
        ]

    skipped = sum(t.skipped for t in tallies.values())
    if skipped:
        lines.append(f"\nskipped {skipped} rows (no reference query, or no database).")

    moved = [c for t in tallies.values() for c in t.changes]
    if moved:
        lines.append(f"\n{len(moved)} grades moved:")
        for c in moved[:40]:
            lines.append(
                f"  {c.strategy:<14} {str(c.question_id):<8} {c.column:<6} {c.direction}"
            )
        if len(moved) > 40:
            lines.append(f"  ... and {len(moved) - 40} more")
    else:
        lines.append("\nNo grade moved. The change does not affect this sweep.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-grade a saved sweep against the current grader. No LLM calls."
    )
    parser.add_argument("--results", required=True, help="Path to a results JSON file.")
    parser.add_argument("--databases", default=None, help="Directory of BIRD .sqlite files.")
    parser.add_argument("--questions", default=None, help="BIRD mini_dev JSON, if not the default.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite the results file with the new grades (default: report only).",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    tallies, updated = regrade(
        results_path,
        databases_root=Path(args.databases) if args.databases else None,
        questions_path=Path(args.questions) if args.questions else None,
    )
    print(report(tallies))

    if args.write:
        # Keep the original once, so a re-grade is reversible and the two can be
        # diffed. Never overwrite an existing backup - that would destroy the
        # only copy of the pre-change grades.
        backup = results_path.with_suffix(results_path.suffix + ".pre-regrade")
        if not backup.exists():
            backup.write_text(results_path.read_text(encoding="utf-8"), encoding="utf-8")
        results_path.write_text(json.dumps(updated, indent=1), encoding="utf-8")
        print(f"\nwrote {results_path}  (previous version kept at {backup.name})")
    else:
        print("\n(report only - pass --write to save the new grades)")


if __name__ == "__main__":
    main()
