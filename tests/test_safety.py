"""Tests for the SQL safety guard.

Each blocked case below is an attack the agent crew could plausibly produce —
either because a model hallucinated it, or because a user asked for it in plain
English and the model complied. The guard is the last line before the database,
so it is tested adversarially rather than happy-path.
"""

from __future__ import annotations

import pytest
import sqlglot

from aqueduct.db.safety import guard


# ── Queries that must be rejected ────────────────────────────────────

BLOCKED = [
    pytest.param("DROP TABLE employees", id="bare-drop"),
    pytest.param("DELETE FROM employees WHERE id = 1", id="delete"),
    pytest.param("UPDATE employees SET salary = 999999", id="update"),
    pytest.param("INSERT INTO employees (id, name) VALUES (99, 'x')", id="insert"),
    pytest.param("TRUNCATE TABLE orders", id="truncate"),
    pytest.param("ALTER TABLE employees ADD COLUMN pwn TEXT", id="alter"),
    pytest.param("CREATE TABLE evil (id INT)", id="create"),
    # Statement chaining — the classic injection shape.
    pytest.param("SELECT 1; DROP TABLE employees", id="chained-drop"),
    pytest.param("SELECT * FROM employees; DELETE FROM orders", id="chained-delete"),
    # A write buried inside an otherwise innocent-looking read.
    pytest.param(
        "WITH gone AS (DELETE FROM employees RETURNING *) SELECT * FROM gone",
        id="write-inside-cte",
    ),
    # Admin and filesystem reach.
    pytest.param("PRAGMA table_info(employees)", id="pragma"),
    pytest.param("ATTACH DATABASE '/etc/passwd' AS pwn", id="attach"),
    pytest.param("SELECT load_extension('evil.so')", id="load-extension"),
    # Not SQL at all — a model returning prose instead of a query.
    pytest.param("I'm sorry, I cannot answer that question.", id="prose-not-sql"),
    pytest.param("", id="empty"),
    pytest.param("   ", id="whitespace-only"),
]


@pytest.mark.parametrize("sql", BLOCKED)
def test_dangerous_sql_is_rejected(sql):
    verdict = guard(sql, dialect="sqlite")
    assert not verdict.ok, f"guard allowed dangerous SQL: {sql!r}"
    assert verdict.reason, "a rejection must explain itself — the Fixer reads this"


# ── Queries that must be allowed ─────────────────────────────────────

ALLOWED = [
    pytest.param("SELECT name FROM employees", id="simple-select"),
    pytest.param(
        "SELECT d.name, COUNT(*) FROM departments d "
        "JOIN employees e ON e.department_id = d.id GROUP BY d.name",
        id="join-with-group-by",
    ),
    pytest.param(
        "WITH t AS (SELECT * FROM employees) SELECT COUNT(*) FROM t",
        id="cte-select",
    ),
    pytest.param(
        "SELECT name FROM employees UNION SELECT city FROM departments",
        id="union",
    ),
    pytest.param(
        "SELECT * FROM employees WHERE salary > (SELECT AVG(salary) FROM employees)",
        id="scalar-subquery",
    ),
]


@pytest.mark.parametrize("sql", ALLOWED)
def test_legitimate_reads_are_allowed(sql):
    verdict = guard(sql, dialect="sqlite")
    assert verdict.ok, f"guard blocked a legitimate read: {verdict.reason}"


# ── Row capping ──────────────────────────────────────────────────────

def test_missing_limit_is_added():
    verdict = guard("SELECT * FROM employees", dialect="sqlite", max_rows=500)
    assert verdict.ok
    assert "LIMIT 500" in verdict.sql


def test_excessive_limit_is_tightened():
    verdict = guard("SELECT * FROM employees LIMIT 100000", dialect="sqlite", max_rows=500)
    assert verdict.ok
    assert "LIMIT 500" in verdict.sql
    assert "100000" not in verdict.sql


def test_modest_limit_is_left_alone():
    verdict = guard("SELECT * FROM employees LIMIT 10", dialect="sqlite", max_rows=500)
    assert verdict.ok
    assert "LIMIT 10" in verdict.sql


# ── Comment handling ─────────────────────────────────────────────────

def test_comments_are_stripped_from_output():
    """Comments carry no value and are one more thing to reason about."""
    verdict = guard("SELECT * FROM employees -- hidden note", dialect="sqlite")
    assert verdict.ok
    assert "hidden note" not in verdict.sql


def test_comment_payload_cannot_break_out():
    """A `*/` inside a comment must not terminate it and start a new statement.

    sqlglot escapes this correctly on its own; the test pins that behaviour so a
    future dependency bump cannot silently regress it.
    """
    payload = "SELECT * FROM employees -- x */ ; DROP TABLE employees"
    verdict = guard(payload, dialect="sqlite")
    assert verdict.ok
    assert len(sqlglot.parse(verdict.sql, dialect="sqlite")) == 1


# ── Output contract ──────────────────────────────────────────────────

def test_emitted_sql_survives_reparsing():
    """Whatever we emit must still be a single read when parsed again."""
    verdict = guard(
        "SELECT d.name, SUM(o.amount) FROM departments d "
        "JOIN employees e ON e.department_id = d.id "
        "JOIN orders o ON o.employee_id = e.id GROUP BY d.name",
        dialect="sqlite",
    )
    assert verdict.ok
    parsed = sqlglot.parse(verdict.sql, dialect="sqlite")
    assert len(parsed) == 1


def test_markdown_fences_are_tolerated():
    """Models wrap SQL in fences out of habit; that alone is not an attack."""
    verdict = guard("```sql\nSELECT name FROM employees\n```", dialect="sqlite")
    assert verdict.ok
    assert "```" not in verdict.sql


def test_verdict_is_falsy_when_blocked():
    assert not guard("DROP TABLE employees", dialect="sqlite")
    assert guard("SELECT 1", dialect="sqlite")
