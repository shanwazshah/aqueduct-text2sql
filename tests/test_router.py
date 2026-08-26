"""Tests for the router and the mechanical schema check.

Both are pure functions over SQL, so they can be tested exhaustively without a
model. That is the point of having built them mechanically.
"""

from __future__ import annotations

import pytest

from aqueduct.agents.critic import check_against_schema
from aqueduct.crew import RepairMode
from aqueduct.db.introspect import load_schema
from aqueduct.router import Router, Tier, score_sql


@pytest.fixture(scope="module")
def schema():
    return load_schema()


# ── risk scoring ─────────────────────────────────────────────────────

SIMPLE = [
    "SELECT name FROM employees",
    "SELECT COUNT(*) FROM employees",
    "SELECT name, city FROM departments",
    "SELECT name FROM employees WHERE salary > 100000",
]

COMPLEX = [
    "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
    "SELECT d.name, SUM(oi.quantity * oi.price) FROM order_items oi "
    "JOIN orders o ON oi.order_id = o.id "
    "JOIN employees e ON o.employee_id = e.id "
    "JOIN departments d ON e.department_id = d.id GROUP BY d.name",
    "SELECT d.name FROM departments d JOIN employees e ON e.department_id = d.id "
    "GROUP BY d.name HAVING AVG(e.salary) > (SELECT AVG(salary) FROM employees)",
    "SELECT e.name, m.name FROM employees e JOIN employees m ON e.manager_id = m.id",
]


@pytest.mark.parametrize("sql", SIMPLE)
def test_simple_queries_are_trusted(sql, schema):
    assert Router(schema=schema).route("q", sql).tier is Tier.TRUST


@pytest.mark.parametrize("sql", COMPLEX)
def test_complex_queries_are_verified(sql, schema):
    assert Router(schema=schema).route("q", sql).tier is Tier.VERIFY


def test_complex_scores_above_simple():
    simple, _ = score_sql("SELECT COUNT(*) FROM employees")
    complex_, _ = score_sql(
        "SELECT d.name FROM departments d JOIN employees e ON e.department_id = d.id "
        "GROUP BY d.name HAVING AVG(e.salary) > (SELECT AVG(salary) FROM employees)"
    )
    assert complex_ > simple


def test_missing_query_is_always_verified():
    """A strategy that produced nothing must never be trusted.

    This is the react failure mode: an empty draft that the repair layer then
    silently rewrote.
    """
    score, signals = score_sql("")
    assert score >= 99
    assert "no query produced" in signals[0]


def test_unparseable_query_is_always_verified():
    score, _ = score_sql("this is not sql")
    assert score >= 99


def test_inner_join_is_not_labelled_outer():
    """Regression: `isinstance(side, str)` matched the empty string.

    Every join was reported as an outer join, inflating scores and printing a
    reason that was not true.
    """
    _, signals = score_sql(
        "SELECT e.name FROM employees e JOIN departments d ON e.department_id = d.id"
    )
    assert "outer join" not in signals


def test_left_join_is_labelled_outer():
    _, signals = score_sql(
        "SELECT e.name FROM employees e LEFT JOIN departments d ON e.department_id = d.id"
    )
    assert "outer join" in signals


def test_subquery_is_not_labelled_a_self_join():
    """Regression: the same table on both sides of a subquery boundary.

    `... WHERE salary > (SELECT AVG(salary) FROM employees)` was reported as a
    self-join because `employees` appears twice in the tree. The tier was right;
    the reason was wrong.
    """
    _, signals = score_sql(
        "SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)"
    )
    assert "self-join" not in signals
    assert any("subquery" in s for s in signals)


def test_real_self_join_is_still_detected():
    _, signals = score_sql(
        "SELECT e.name, m.name FROM employees e JOIN employees m ON e.manager_id = m.id"
    )
    assert "self-join" in signals


def test_verdict_explains_itself():
    verdict = Router().route("q", "SELECT COUNT(*) FROM employees")
    assert verdict.explain()


def test_tiers_map_to_repair_modes():
    assert Tier.TRUST.repair_mode is RepairMode.EXECUTION
    assert Tier.VERIFY.repair_mode is RepairMode.BOTH


# ── schema check: false positives that cost real money ───────────────
#
# Each of these fired on a *correct* query under the regex implementation. Every
# one would have triggered a pointless repair, and through the router, spent a
# critic call reviewing SQL that had nothing wrong with it.

def test_output_alias_is_not_a_missing_column(schema):
    assert check_against_schema(
        "SELECT SUM(amount) AS total_value FROM orders WHERE status = 'shipped'", schema
    ) == []


def test_string_literal_is_not_a_missing_column(schema):
    assert check_against_schema(
        "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'", schema
    ) == []


def test_type_name_and_function_are_not_missing_columns(schema):
    assert check_against_schema(
        "SELECT CAST(SUM(CASE WHEN status = 'cancelled' THEN 1 ELSE 0 END) AS REAL) "
        "* 100 / COUNT(*) FROM orders",
        schema,
    ) == []


def test_cte_name_is_not_a_missing_table(schema):
    assert check_against_schema(
        "WITH recent AS (SELECT * FROM orders) SELECT COUNT(*) FROM recent", schema
    ) == []


def test_self_join_alias_resolves(schema):
    assert check_against_schema(
        "SELECT e.name, m.name FROM employees e JOIN employees m ON e.manager_id = m.id",
        schema,
    ) == []


# ── ...while still catching the real thing ───────────────────────────

@pytest.mark.parametrize(
    "sql,needle",
    [
        ("SELECT dept FROM employees", "dept"),
        ("SELECT e.department FROM employees e", "department"),
        ("SELECT * FROM staff", "staff"),
        ("SELECT salery FROM employees", "salary"),      # suggestion
        ("SELECT o.total FROM orders o", "total"),
    ],
)
def test_real_hallucinations_are_still_caught(sql, needle, schema):
    errors = check_against_schema(sql, schema)
    assert errors, f"missed a hallucination in: {sql}"
    assert any(needle in e for e in errors)
