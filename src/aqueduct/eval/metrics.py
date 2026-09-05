"""Grading.

The headline metric is **execution accuracy (EX)**: run the generated query and
the reference query, and compare the result sets. That is the standard for
Text-to-SQL benchmarks including BIRD and Spider, and it is the only fair
comparison — `SELECT COUNT(*)` and `SELECT COUNT(id)` are different strings and
the same answer.

Two judgement calls, both of which matter and neither of which is obvious:

  * **Column order is ignored, row order is not — unless the question never
    asked for an order.** A reference query with `ORDER BY` is making a claim
    about sequence, and a prediction is held to it whether or not it sorts. When
    the reference does not sort, the database is free to return rows however it
    likes, so comparing as sets is the only stable choice — and a prediction that
    sorts anyway is not punished for it.
  * **Numbers are compared with a tolerance.** `AVG` over floats can differ in
    the last bits between two algebraically identical queries. Failing a correct
    query over 1e-9 would be measuring floating point, not SQL.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..db.engine import QueryResult, run_query

FLOAT_TOLERANCE = 1e-6


@dataclass
class Grade:
    """The outcome of grading one question."""

    correct: bool
    reason: str
    predicted_rows: int = 0
    gold_rows: int = 0

    def __bool__(self) -> bool:
        return self.correct


def normalise_value(value):
    """Make a cell comparable across queries.

    Floats are rounded so tolerance works via plain equality, and everything
    else is compared as a string — SQLite is loosely typed, so the same value
    can arrive as `1` from one query and `'1'` from another.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        rounded = round(float(value), 6)
        # 4.0 and 4 are the same answer; normalise so they compare equal.
        return int(rounded) if rounded == int(rounded) else rounded
    return str(value).strip()


def normalise_rows(result: QueryResult, ordered: bool):
    """Turn a result into a comparable structure.

    Cells are compared **positionally**, matching BIRD's and Spider's official
    evaluation. Only row order is relaxed, and only when neither query asked for
    an order.

    This was not the first implementation. Two earlier attempts tried to ignore
    column order, on the reasoning that "name, count" and "count, name" answer
    the same question:

      1. sorting cells within each row — collapses `(min=5, max=10)` and
         `(min=10, max=5)` to the same tuple;
      2. sorting whole columns into a canonical order — same failure, because
         for a single-row result those two answers *are* the same multiset.

    The second attempt is the interesting one: the problem is not the algorithm,
    it is the goal. When column names are ignored, column order carries the only
    information distinguishing those two answers, so any scheme that discards it
    must score them equal. Column names cannot rescue it either, since a model
    writing `AVG(salary)` where the reference writes `avg_sal` would then fail
    for naming rather than for being wrong.

    So the relaxation was dropped. Positional comparison is what the published
    benchmarks do, it cannot produce a false pass, and matching it is what makes
    our numbers comparable to theirs. The regression test stays as a guard.
    """
    rows = [tuple(str(normalise_value(v)) for v in row) for row in result.rows]
    return rows if ordered else sorted(rows)


def wants_order(sql: str) -> bool:
    """Does this query claim a specific row order?"""
    return "order by" in sql.lower()


def execution_accuracy(
    predicted_sql: str,
    gold_sql: str,
    db_url: str | None = None,
) -> Grade:
    """Compare a generated query against a reference by running both."""
    if not predicted_sql or not predicted_sql.strip():
        return Grade(False, "no query produced")

    gold = run_query(gold_sql, db_url=db_url)
    if not gold.ok:
        # The reference itself is broken — a bug in the question set, not in the
        # agent. Surfaced loudly rather than silently scored as a failure.
        return Grade(False, f"REFERENCE QUERY FAILED: {gold.error}")

    predicted = run_query(predicted_sql, db_url=db_url)
    if not predicted.ok:
        return Grade(False, f"query failed: {predicted.error}", gold_rows=gold.row_count)

    # Order-sensitivity is a property of the *reference*, not of the prediction.
    # This used to be `wants_order(gold) and wants_order(predicted)`, which let a
    # prediction opt out of the check simply by omitting ORDER BY: a gold query
    # sorting by salary and a prediction with no sort at all returned the same
    # rows in a different order and scored `match`. The error ran in the
    # flattering direction, which is the one to be suspicious of.
    ordered = wants_order(gold_sql)
    got = normalise_rows(predicted, ordered)
    want = normalise_rows(gold, ordered)

    if got == want:
        return Grade(True, "match", predicted.row_count, gold.row_count)

    if predicted.row_count != gold.row_count:
        return Grade(
            False,
            f"row count differs: got {predicted.row_count}, expected {gold.row_count}",
            predicted.row_count,
            gold.row_count,
        )

    return Grade(
        False,
        "same row count, different values",
        predicted.row_count,
        gold.row_count,
    )


@dataclass
class Report:
    """Aggregate results across a question set."""

    total: int = 0
    correct: int = 0
    failed_to_run: int = 0
    by_difficulty: dict[str, tuple[int, int]] = None  # level -> (correct, total)
    by_trap: dict[str, tuple[int, int]] = None
    seconds: float = 0.0
    llm_calls: int = 0

    def __post_init__(self):
        self.by_difficulty = self.by_difficulty or {}
        self.by_trap = self.by_trap or {}

    @property
    def ex(self) -> float:
        """Execution accuracy as a percentage."""
        return 100.0 * self.correct / self.total if self.total else 0.0

    def add(self, grade: Grade, difficulty: str, trap: str | None) -> None:
        self.total += 1
        if grade.correct:
            self.correct += 1
        elif "query failed" in grade.reason:
            self.failed_to_run += 1

        c, t = self.by_difficulty.get(difficulty, (0, 0))
        self.by_difficulty[difficulty] = (c + int(grade.correct), t + 1)

        if trap:
            c, t = self.by_trap.get(trap, (0, 0))
            self.by_trap[trap] = (c + int(grade.correct), t + 1)

    def render(self) -> str:
        lines = [
            f"EX: {self.ex:.1f}%  ({self.correct}/{self.total})",
            f"failed to run: {self.failed_to_run}",
            f"time: {self.seconds:.1f}s   llm calls: {self.llm_calls}",
        ]
        if self.by_difficulty:
            lines.append("\nby difficulty:")
            for level in ("easy", "medium", "hard"):
                if level in self.by_difficulty:
                    c, t = self.by_difficulty[level]
                    lines.append(f"  {level:<8} {c}/{t}")
        if self.by_trap:
            lines.append("\nby trap:")
            for trap, (c, t) in sorted(self.by_trap.items()):
                lines.append(f"  {trap:<18} {c}/{t}")
        return "\n".join(lines)
