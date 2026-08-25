"""Tests for the mechanical half of the Critic, and for error memory.

Neither of these touches an LLM. That is the point: the checks worth testing
deterministically are exactly the ones that should never have been given to a
model in the first place.
"""

from __future__ import annotations

import pytest

from aqueduct.agents.critic import check_against_schema
from aqueduct.agents.memory import ErrorMemory
from aqueduct.db.introspect import load_schema


@pytest.fixture(scope="module")
def schema():
    return load_schema()


# ── hallucination detection ──────────────────────────────────────────

def test_missing_table_is_caught(schema):
    errors = check_against_schema("SELECT * FROM staff", schema)
    assert any("staff" in e for e in errors)


def test_missing_qualified_column_is_caught(schema):
    errors = check_against_schema("SELECT e.department FROM employees e", schema)
    assert any("department" in e for e in errors)


def test_missing_unqualified_column_is_caught(schema):
    errors = check_against_schema("SELECT dept FROM employees", schema)
    assert any("dept" in e for e in errors)


def test_suggestion_points_at_the_right_column(schema):
    """The hint is the useful part.

    'no such column: salery' tells the Fixer something is wrong; 'did you mean
    salary' tells it what to write.
    """
    errors = check_against_schema("SELECT salery FROM employees", schema)
    assert any("salary" in e for e in errors)


def test_suggestions_are_scoped_to_the_queried_table(schema):
    """Regression: candidates were drawn from every table in the database.

    'dept' against the whole schema matched 'budget' — a column on a different
    table entirely, and a hint that would send the Fixer somewhere useless.
    """
    errors = check_against_schema("SELECT dept FROM employees", schema)
    assert not any("budget" in e for e in errors)


# ── queries that must pass cleanly ───────────────────────────────────

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT name, salary FROM employees",
        "SELECT e.name, d.name FROM employees e JOIN departments d ON e.department_id = d.id",
        "SELECT COUNT(*) FROM orders WHERE status = 'shipped'",
        "SELECT p.category, SUM(oi.quantity * oi.price) FROM order_items oi "
        "JOIN products p ON oi.product_id = p.id GROUP BY p.category",
    ],
)
def test_valid_queries_produce_no_errors(sql, schema):
    assert check_against_schema(sql, schema) == []


def test_aggregate_keywords_are_not_mistaken_for_columns(schema):
    """COUNT, SUM and friends are not hallucinated column names."""
    assert check_against_schema("SELECT COUNT(*), MAX(salary) FROM employees", schema) == []


# ── error memory ─────────────────────────────────────────────────────

@pytest.fixture
def memory(tmp_path):
    return ErrorMemory(tmp_path / "memory.json")


def test_lesson_is_stored_and_recalled(memory):
    memory.record(
        "How many staff work in Engineering?",
        "SELECT COUNT(*) FROM staff",
        "table 'staff' does not exist",
        "SELECT COUNT(*) FROM employees",
    )
    assert len(memory) == 1
    assert memory.recall("How many staff work in Engineering?")


def test_unrelated_questions_recall_nothing(memory):
    """An irrelevant example in the prompt is a distraction, not a hint."""
    memory.record(
        "How many staff in Engineering?",
        "SELECT COUNT(*) FROM staff",
        "table 'staff' does not exist",
        "SELECT COUNT(*) FROM employees",
    )
    assert memory.recall("What is the capital of France?") == []


def test_topically_adjacent_lesson_is_not_recalled(memory):
    """Regression: this cost a correct answer.

    A lesson about "employees who never handled an order" was recalled for
    "revenue by department", because both mention orders and employees. The
    Writer copied the corrected query's join shape into an unrelated question
    and turned a passing answer into a failing one.

    Sharing a keyword is not relevance. The threshold is what stops it.
    """
    memory.record(
        "Which employees have never handled an order?",
        "SELECT e.name FROM employees e LEFT JOIN order_items oi ON e.id = oi.employee_id",
        "no such column: oi.employee_id",
        "SELECT e.name FROM employees e LEFT JOIN orders o ON e.id = o.employee_id",
    )
    unrelated = (
        "What is the total revenue from order items, broken down by the "
        "department of the salesperson who handled the order?"
    )
    assert memory.recall(unrelated) == []


def test_rendered_lesson_contains_no_sql(memory):
    """Lessons are rules, not examples.

    Rendering the corrected query in full caused the Writer to imitate its
    shape on questions where that shape was wrong.
    """
    memory.record(
        "Which employees have never handled an order?",
        "SELECT e.name FROM employees e LEFT JOIN order_items oi ON e.id = oi.employee_id",
        "no such column: oi.employee_id",
        "SELECT e.name FROM employees e LEFT JOIN orders o ON e.id = o.employee_id",
    )
    rendered = memory.render_for_prompt("Which employees have never handled an order?")
    assert rendered
    assert "SELECT" not in rendered
    assert "LEFT JOIN" not in rendered


def test_noop_fix_is_not_stored(memory):
    """A 'repair' that changed nothing teaches nothing."""
    memory.record("q", "SELECT 1", "some error", "SELECT 1")
    assert len(memory) == 0


def test_duplicate_lesson_is_not_stored_twice(memory):
    for _ in range(3):
        memory.record(
            "How many staff?",
            "SELECT COUNT(*) FROM staff",
            "table 'staff' does not exist",
            "SELECT COUNT(*) FROM employees",
        )
    assert len(memory) == 1


def test_memory_survives_reload(memory, tmp_path):
    memory.record(
        "How many staff?",
        "SELECT COUNT(*) FROM staff",
        "table 'staff' does not exist",
        "SELECT COUNT(*) FROM employees",
    )
    assert len(ErrorMemory(tmp_path / "memory.json")) == 1


def test_corrupt_memory_file_does_not_crash(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert len(ErrorMemory(path)) == 0


def test_prompt_rendering_is_empty_when_nothing_relevant(memory):
    assert memory.render_for_prompt("anything at all") == ""
