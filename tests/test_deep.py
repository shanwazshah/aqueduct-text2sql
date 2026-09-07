"""Tests for the Phase 9 deep agent and its cost-matched control.

Everything here runs without a model. The agent's loop needs one, but the parts
worth testing do not: the join finder and the value prober are the mechanical
sub-agents that exist precisely so a model is not asked a decidable question, and
the self-consistency vote is arithmetic over result sets.

`StrategyContext.client` is stubbed where a model would otherwise be called, so
the voting logic is exercised on scripted samples rather than on whatever the 3B
model happens to say today.
"""

from __future__ import annotations

import pytest

from aqueduct.db.introspect import load_schema
from aqueduct.llm.client import Usage
from aqueduct.observability.trace import Trace
from aqueduct.strategies.base import StrategyContext
from aqueduct.strategies.deep import (
    Action,
    DeepAgentStrategy,
    DeepSeededStrategy,
    Scratchpad,
    find_values,
    join_path,
)
from aqueduct.strategies.self_consistency import SelfConsistencyStrategy, _answer_key


@pytest.fixture(scope="module")
def schema():
    return load_schema()


@pytest.fixture
def ctx(schema):
    return StrategyContext(
        question="Which department has the highest total salary spend?",
        schema=schema,
        trace=Trace("q"),
        usage=Usage(),
    )


# ── the join finder ──────────────────────────────────────────────────

def test_a_direct_foreign_key_is_found(schema):
    path = join_path(schema, "employees", "departments")
    assert path == ["employees.department_id = departments.id"]


def test_the_path_is_symmetric(schema):
    there = join_path(schema, "employees", "departments")
    back = join_path(schema, "departments", "employees")
    assert there == back, "a foreign key joins two tables regardless of direction"


def test_a_multi_hop_path_is_found(schema):
    """products and orders connect only through order_items."""
    path = join_path(schema, "products", "orders")
    assert len(path) == 2
    assert any("order_items" in c and "products" in c for c in path)
    assert any("order_items" in c and "orders" in c for c in path)


def test_an_unconnected_pair_returns_nothing(schema):
    assert join_path(schema, "employees", "not_a_table") == []


def test_a_table_joined_to_itself_returns_nothing(schema):
    """Not an error - there is simply no path to find."""
    assert join_path(schema, "employees", "employees") == []


def test_the_conditions_are_read_from_the_schema_not_guessed(schema):
    """Every column named in a returned condition must actually exist."""
    for condition in join_path(schema, "products", "orders"):
        left, right = condition.split(" = ")
        for side in (left, right):
            table, column = side.split(".")
            match = [t for t in schema.tables if t.name == table]
            assert match, f"{table} is not a table"
            assert any(c.name == column for c in match[0].columns), condition


# ── the value prober ─────────────────────────────────────────────────

def test_values_come_back_for_a_real_column(schema):
    values = find_values(schema, "orders", "status")
    assert values
    assert all(isinstance(v, str) for v in values)


def test_a_close_spelling_is_ranked_first(schema):
    """The failure this exists for: the question says Shipped, the column holds shipped."""
    values = find_values(schema, "orders", "status", like="Shipped")
    assert values[0].lower() == "shipped"


def test_an_unknown_column_returns_nothing_rather_than_erroring(schema):
    assert find_values(schema, "orders", "not_a_column") == []
    assert find_values(schema, "not_a_table", "status") == []


# ── the scratchpad ───────────────────────────────────────────────────

def test_notes_do_not_grow_without_bound():
    """The property that keeps step eight as cheap as step one."""
    pad = Scratchpad()
    for i in range(50):
        pad.note(pad.join_notes, f"note {i}")
    assert len(pad.join_notes) <= 8
    assert "note 49" in pad.join_notes, "the most recent note must survive"


def test_a_repeated_note_is_not_stored_twice():
    pad = Scratchpad()
    pad.note(pad.value_notes, "orders.status: 'shipped'")
    pad.note(pad.value_notes, "orders.status: 'shipped'")
    assert len(pad.value_notes) == 1


def test_an_empty_scratchpad_renders_something_usable():
    assert Scratchpad().render() == "(nothing learned yet)"


def test_plan_progress_is_rendered():
    pad = Scratchpad(plan=["find the tables", "join them", "aggregate"], done=1)
    rendered = pad.render()
    assert "[x] 1. find the tables" in rendered
    assert "[ ] 2. join them" in rendered


# ── the agent's tools, driven directly ───────────────────────────────

def agent():
    return DeepAgentStrategy()


def test_inspect_tables_records_what_it_saw(ctx):
    pad = Scratchpad()
    out, ran = agent()._dispatch(ctx, pad, Action(tool="inspect_tables", tables=["employees"]))
    assert not ran
    assert "employees" in out
    assert pad.schema_notes and "salary" in pad.schema_notes[0]


def test_inspect_tables_names_the_alternatives_when_wrong(ctx):
    out, _ = agent()._dispatch(ctx, Scratchpad(), Action(tool="inspect_tables", tables=["staff"]))
    assert "No such table" in out
    assert "employees" in out


def test_find_join_path_notes_the_exact_condition(ctx):
    pad = Scratchpad()
    out, _ = agent()._dispatch(
        ctx, pad, Action(tool="find_join_path", from_table="employees", to_table="departments")
    )
    assert out == "employees.department_id = departments.id"
    assert pad.join_notes == [out]


def test_find_join_path_crosses_three_hops(ctx):
    """employees -> orders -> order_items -> products, none of it guessed."""
    pad = Scratchpad()
    out, _ = agent()._dispatch(
        ctx, pad, Action(tool="find_join_path", from_table="employees", to_table="products")
    )
    assert out.count(" AND ") == 2
    assert "order_items.product_id = products.id" in out


def test_find_join_path_says_so_when_there_is_none(ctx):
    """A disconnected pair, on a schema built to have one.

    The demo database is fully connected, so this needs its own fixture - and
    "no path" is information rather than an error: it means the question needs a
    table nobody has mentioned.
    """
    from aqueduct.db.introspect import Column, ForeignKey, Schema, Table

    island = Schema([
        Table("a", [Column("id", "INTEGER", False, True)],
              [ForeignKey("id", "b", "id")]),
        Table("b", [Column("id", "INTEGER", False, True)]),
        Table("lonely", [Column("id", "INTEGER", False, True)]),
    ])
    assert join_path(island, "a", "b") == ["a.id = b.id"]
    assert join_path(island, "a", "lonely") == []

    pad = Scratchpad()
    ctx.schema = island
    out, _ = agent()._dispatch(
        ctx, pad, Action(tool="find_join_path", from_table="a", to_table="lonely")
    )
    assert "No foreign-key path" in out
    assert pad.join_notes


def test_try_sql_reports_rows(ctx):
    pad = Scratchpad()
    out, ran = agent()._dispatch(
        ctx, pad, Action(tool="try_sql", sql="SELECT COUNT(*) FROM employees")
    )
    assert ran
    assert "rows:" in out
    assert pad.attempts


def test_try_sql_reports_the_database_error(ctx):
    out, ran = agent()._dispatch(
        ctx, Scratchpad(), Action(tool="try_sql", sql="SELECT nope FROM employees")
    )
    assert not ran
    assert out.startswith("ERROR:")


def test_an_empty_result_is_flagged_as_suspicious(ctx):
    """Valid SQL and zero rows is the failure with no error to learn from."""
    out, ran = agent()._dispatch(
        ctx, Scratchpad(),
        Action(tool="try_sql", sql="SELECT * FROM orders WHERE status = 'Shipped'"),
    )
    assert ran
    assert "0 rows" in out and "check the values" in out


def test_the_two_variants_are_distinct_and_named_correctly():
    assert DeepAgentStrategy.name == "deep" and DeepAgentStrategy.seeded is False
    assert DeepSeededStrategy.name == "deep_seeded" and DeepSeededStrategy.seeded is True


# ── the self-consistency control ─────────────────────────────────────

class StubClient:
    """Returns scripted SQL, one sample per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        return self.replies.pop(0) if self.replies else ""


def vote_with(ctx, monkeypatch, replies):
    stub = StubClient(replies)
    monkeypatch.setattr(StrategyContext, "client", lambda self, role, temperature=None: stub)
    draft = SelfConsistencyStrategy(samples=len(replies)).generate(ctx)
    return draft, stub


def test_the_majority_answer_wins(ctx, monkeypatch):
    draft, _ = vote_with(ctx, monkeypatch, [
        "SELECT COUNT(*) FROM employees",
        "SELECT COUNT(*) FROM employees",
        "SELECT COUNT(*) FROM departments",
    ])
    assert draft.sql == "SELECT COUNT(*) FROM employees"
    assert draft.notes["agreement"] == 2
    assert draft.notes["distinct_answers"] == 2


def test_queries_that_differ_in_text_but_agree_on_the_answer_count_together(ctx, monkeypatch):
    """The reason voting is on the result set and not on the SQL."""
    draft, _ = vote_with(ctx, monkeypatch, [
        "SELECT COUNT(*) FROM employees",
        "SELECT COUNT(id) FROM employees",
        "SELECT COUNT(*) FROM departments",
    ])
    assert draft.notes["distinct_answers"] == 2
    assert draft.notes["agreement"] == 2
    assert "employees" in draft.sql


def test_a_query_that_cannot_run_gets_no_vote(ctx, monkeypatch):
    draft, _ = vote_with(ctx, monkeypatch, [
        "SELECT nope FROM employees",
        "SELECT nope FROM employees",
        "SELECT COUNT(*) FROM employees",
    ])
    assert draft.sql == "SELECT COUNT(*) FROM employees"
    assert draft.notes["agreement"] == 1


def test_when_nothing_runs_the_first_draft_is_returned_with_no_agreement(ctx, monkeypatch):
    draft, _ = vote_with(ctx, monkeypatch, ["SELECT nope FROM employees"] * 3)
    assert draft.sql == "SELECT nope FROM employees"
    assert draft.notes["agreement"] == 0, "a score must not read as consensus when there was none"


def test_it_actually_spends_the_samples_it_claims(ctx, monkeypatch):
    """The whole point is cost parity with the deep agent. Silently sampling once
    would make it look free and measure nothing."""
    _, stub = vote_with(ctx, monkeypatch, ["SELECT COUNT(*) FROM employees"] * 5)
    assert stub.calls == 5


def test_unanimity_is_recorded():
    assert SelfConsistencyStrategy(samples=3).samples == 3


def test_answer_keys_ignore_row_order():
    class R:
        rows = [(1, "a"), (2, "b")]

    class S:
        rows = [(2, "b"), (1, "a")]

    assert _answer_key(R()) == _answer_key(S())


def test_answer_keys_separate_genuinely_different_answers():
    class R:
        rows = [(1,)]

    class S:
        rows = [(2,)]

    assert _answer_key(R()) != _answer_key(S())
