"""Tests for the grader.

A grader that scores wrong answers as correct is worse than no grader, because
every number downstream of it is quietly false. These tests exist mostly to pin
down the cases where "close enough" is and is not acceptable.
"""

from __future__ import annotations

import pytest

from aqueduct.db.engine import QueryResult
from aqueduct.eval.metrics import execution_accuracy, normalise_rows, normalise_value


def make(columns: list[str], rows: list[tuple]) -> QueryResult:
    return QueryResult(ok=True, sql="", columns=columns, rows=rows)


# ── value normalisation ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "a,b",
    [
        (4, 4.0),               # COUNT(*) vs SUM(1)
        (1.0000001, 1.0),       # float drift within tolerance
        ("Berlin", " Berlin "), # incidental whitespace
    ],
)
def test_equivalent_values_normalise_equal(a, b):
    assert normalise_value(a) == normalise_value(b)


def test_distinct_values_stay_distinct():
    assert normalise_value(4) != normalise_value(5)
    assert normalise_value("Berlin") != normalise_value("Prague")


# ── column ordering ──────────────────────────────────────────────────

def test_column_order_matters():
    """Positional comparison, as BIRD and Spider both do.

    Two earlier versions of the grader tried to ignore column order. Both
    produced false passes, because once column order is discarded there is
    nothing left to distinguish `(5, 10)` from `(10, 5)`.
    """
    a = make(["name", "n"], [("Engineering", 4)])
    b = make(["n", "name"], [(4, "Engineering")])
    assert normalise_rows(a, ordered=False) != normalise_rows(b, ordered=False)


def test_swapped_values_are_not_treated_as_equal():
    """Regression guard for the false pass that killed the clever version."""
    a = make(["min", "max"], [(5, 10)])
    b = make(["min", "max"], [(10, 5)])
    assert normalise_rows(a, ordered=False) != normalise_rows(b, ordered=False)


# ── row ordering ─────────────────────────────────────────────────────

def test_row_order_ignored_when_unordered():
    a = make(["name"], [("Alice",), ("Bob",)])
    b = make(["name"], [("Bob",), ("Alice",)])
    assert normalise_rows(a, ordered=False) == normalise_rows(b, ordered=False)


def test_row_order_respected_when_ordered():
    a = make(["name"], [("Alice",), ("Bob",)])
    b = make(["name"], [("Bob",), ("Alice",)])
    assert normalise_rows(a, ordered=True) != normalise_rows(b, ordered=True)


# ── end to end, against the real demo database ───────────────────────

def test_equivalent_queries_score_correct():
    grade = execution_accuracy(
        "SELECT COUNT(id) FROM employees",
        "SELECT COUNT(*) FROM employees",
    )
    assert grade.correct, grade.reason


def test_wrong_query_scores_incorrect():
    grade = execution_accuracy(
        "SELECT COUNT(*) FROM departments",
        "SELECT COUNT(*) FROM employees",
    )
    assert not grade.correct


def test_extra_column_is_scored_incorrect():
    """Strict, and deliberately so.

    Returning `name, salary` when the reference returns `name` is arguably more
    useful, but BIRD and Spider both compare full result sets. Relaxing this
    would inflate our numbers relative to published baselines and make the
    comparison meaningless.
    """
    grade = execution_accuracy(
        "SELECT name, salary FROM employees WHERE salary > 100000",
        "SELECT name FROM employees WHERE salary > 100000",
    )
    assert not grade.correct


def test_broken_reference_is_reported_loudly():
    """A bad gold query is a bug in the question set, not an agent failure."""
    grade = execution_accuracy(
        "SELECT 1",
        "SELECT * FROM table_that_does_not_exist",
    )
    assert not grade.correct
    assert "REFERENCE QUERY FAILED" in grade.reason


def test_unrunnable_prediction_is_scored_not_crashed():
    grade = execution_accuracy(
        "SELECT nonexistent_column FROM employees",
        "SELECT COUNT(*) FROM employees",
    )
    assert not grade.correct
    assert "query failed" in grade.reason
