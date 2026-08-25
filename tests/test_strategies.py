"""Tests for the strategy layer.

No LLM calls here. These cover the contract every strategy must satisfy and the
parsing helpers around it — the parts that can break silently and would then
show up as a mysterious accuracy drop in a two-hour sweep rather than as an
error.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aqueduct.llm.types import UnitFloat, _to_unit_interval
from aqueduct.strategies import STRATEGIES, get_strategy
from aqueduct.strategies.base import Strategy, strip_sql
from aqueduct.strategies.eval_optimize import Evaluation


# ── the registry ─────────────────────────────────────────────────────

def test_every_registered_strategy_builds():
    for name in STRATEGIES:
        assert isinstance(get_strategy(name), Strategy)


def test_strategy_names_match_their_registry_key():
    """A mismatch here would mislabel every row of the leaderboard."""
    for name in STRATEGIES:
        assert get_strategy(name).name == name


def test_unknown_strategy_is_rejected():
    with pytest.raises(KeyError):
        get_strategy("does-not-exist")


def test_all_six_strategies_are_registered():
    assert set(STRATEGIES) == {
        "direct", "react", "chain", "parallel", "eval_optimize", "orchestrator",
    }


# ── SQL extraction ───────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SELECT 1", "SELECT 1"),
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("SQL: SELECT 1", "SELECT 1"),
        ("sql: SELECT 1", "SELECT 1"),
        ("  SELECT 1  ", "SELECT 1"),
        ("Query: SELECT 1", "SELECT 1"),
    ],
)
def test_strip_sql_handles_model_habits(raw, expected):
    assert strip_sql(raw) == expected


# ── bounded numbers from models ──────────────────────────────────────

@pytest.mark.parametrize(
    "given,expected",
    [
        (0.9, 0.9),      # already correct
        (100, 1.0),      # model answered in percent
        (85, 0.85),      # ditto
        (1, 1.0),        # ambiguous, but 1.0 either way
        (-5, 0.0),       # clamped
        (999, 1.0),      # clamped
        (True, 1.0),     # boolean instead of a number
    ],
)
def test_confidence_is_coerced_into_range(given, expected):
    """Regression: a run died because a 0.0-1.0 field came back as 100.

    Constrained decoding enforces the JSON shape, not the value range - the
    schema's `minimum`/`maximum` are validation keywords the sampler ignores.
    """
    assert _to_unit_interval(given) == pytest.approx(expected)


def test_unit_float_accepts_out_of_range_input_through_pydantic():
    from pydantic import BaseModel

    class Model(BaseModel):
        score: UnitFloat

    assert Model(score=100).score == 1.0
    assert Model(score=0.5).score == 0.5


def test_non_numeric_confidence_still_raises():
    """Coercion is for scale confusion, not for silently swallowing junk."""
    from pydantic import BaseModel

    class Model(BaseModel):
        score: UnitFloat

    with pytest.raises(ValidationError):
        Model(score="not a number")


# ── evaluator scoring ────────────────────────────────────────────────

def test_evaluation_passes_only_when_every_criterion_is_top_marked():
    perfect = Evaluation(
        answers_question=3, joins_correct=3, aggregation_correct=3, filters_correct=3
    )
    assert perfect.passes()

    one_weak = Evaluation(
        answers_question=3, joins_correct=2, aggregation_correct=3, filters_correct=3
    )
    assert not one_weak.passes()


def test_evaluation_identifies_its_weakest_area():
    evaluation = Evaluation(
        answers_question=3, joins_correct=0, aggregation_correct=2, filters_correct=3
    )
    assert evaluation.weakest == "joins"
    assert evaluation.total == 8


# ── tool-call recovery ───────────────────────────────────────────────
#
# `qwen2.5-coder:3b` advertises tool support and emits correct tool calls as
# plain text in `content`, leaving `tool_calls` empty. The ReAct loop saw no
# calls, exited immediately, returned an empty query, and the repair layer
# silently wrote every query instead — producing a completely false 95.5% for
# the strategy on the first sweep.

from types import SimpleNamespace

from aqueduct.strategies.react import extract_tool_calls


def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def test_native_tool_calls_are_used_when_present():
    message = _message(
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                function=SimpleNamespace(name="list_tables", arguments="{}"),
            )
        ]
    )
    calls = extract_tool_calls(message)
    assert len(calls) == 1
    assert calls[0].name == "list_tables"
    assert calls[0].native


def test_tool_call_written_as_text_is_recovered():
    """The exact shape qwen2.5-coder emits."""
    calls = extract_tool_calls(_message(content='{"name": "list_tables", "arguments": {}}'))
    assert len(calls) == 1
    assert calls[0].name == "list_tables"
    assert not calls[0].native


def test_recovered_call_keeps_its_arguments():
    calls = extract_tool_calls(
        _message(content='{"name": "get_table_schema", "arguments": {"table_name": "orders"}}')
    )
    assert calls[0].arguments == {"table_name": "orders"}


@pytest.mark.parametrize(
    "content",
    [
        '<tool_call>{"name": "list_tables", "arguments": {}}</tool_call>',
        '```json\n{"name": "list_tables", "arguments": {}}\n```',
        '[{"name": "list_tables", "arguments": {}}]',
        '{"function": {"name": "list_tables", "arguments": {}}}',
        '{"name": "list_tables", "arguments": "{}"}',
    ],
)
def test_tool_call_wrappers_are_tolerated(content):
    calls = extract_tool_calls(_message(content=content))
    assert [c.name for c in calls] == ["list_tables"]


@pytest.mark.parametrize(
    "content",
    ["", None, "I will now list the tables.", "not json at all", "{broken json"],
)
def test_plain_prose_is_not_mistaken_for_a_tool_call(content):
    """A final answer must not be parsed as a call, or the loop never ends."""
    assert extract_tool_calls(_message(content=content)) == []
