"""Tests for the Phase 8 memory arm.

The measurement this arm produces is "memory on versus memory off". Three things
have to hold for that number to mean anything, and none of them held before:

  * the arms must not share a memory, or a lesson `direct` learned improves
    `chain` and the comparison is between two contaminated runs;
  * the model control must not inherit the previous model's lessons;
  * a result of +0.0 has to be distinguishable from a memory that never fired,
    which needs the recall count recorded next to the score.

No LLM is called here. `RepairMode.NONE` with a stub strategy exercises the whole
`Crew.ask` path without a single model request, so these run in CI and in a
second.
"""

from __future__ import annotations

from pathlib import Path

from aqueduct.agents.memory import ErrorMemory
from aqueduct.crew import Crew, RepairMode
from aqueduct.eval.bird_run import Row, memory_path_for
from aqueduct.strategies.base import Draft, Strategy


class StubStrategy(Strategy):
    """Returns fixed SQL. Keeps generation out of the picture entirely."""

    name = "stub"

    def __init__(self, sql: str):
        self.sql = sql

    def generate(self, ctx) -> Draft:
        return Draft(sql=self.sql, notes={"calls": 0})


def crew_with(memory: ErrorMemory | None, *, use_memory: bool) -> Crew:
    return Crew(
        strategy=StubStrategy("SELECT COUNT(*) FROM employees"),
        repair=RepairMode.NONE,
        memory=memory,
        use_memory=use_memory,
    )


LESSON = dict(
    question="How many employees are in the sales department?",
    broken_sql="SELECT COUNT(*) FROM employees WHERE dept = 'Sales'",
    error="no such column: dept",
    fixed_sql="SELECT COUNT(*) FROM employees WHERE department_id = 3",
)


# ── the recall count ─────────────────────────────────────────────────

def test_a_recalled_lesson_is_counted(tmp_path):
    """Without this count, +0.0 cannot be told from "never fired"."""
    memory = ErrorMemory(tmp_path / "m.json")
    memory.record(**LESSON)

    answer = crew_with(memory, use_memory=True).ask(
        "How many employees are in the sales department?"
    )
    assert answer.lessons_recalled == 1


def test_an_irrelevant_lesson_is_not_counted(tmp_path):
    memory = ErrorMemory(tmp_path / "m.json")
    memory.record(**LESSON)

    answer = crew_with(memory, use_memory=True).ask(
        "What is the average unit cost of products by category?"
    )
    assert answer.lessons_recalled == 0


def test_memory_off_recalls_nothing_even_with_lessons_stored(tmp_path):
    """The off arm must be genuinely off, not merely empty."""
    memory = ErrorMemory(tmp_path / "m.json")
    memory.record(**LESSON)

    answer = crew_with(memory, use_memory=False).ask(
        "How many employees are in the sales department?"
    )
    assert answer.lessons_recalled == 0


def test_no_memory_configured_is_zero_not_a_crash():
    answer = crew_with(None, use_memory=False).ask("How many employees are there?")
    assert answer.lessons_recalled == 0
    assert answer.ok


# ── arm isolation ────────────────────────────────────────────────────

def test_each_strategy_gets_its_own_memory_file():
    results = Path("data/bird/bird_results_7b.json")
    direct = memory_path_for(results, "direct")
    chain = memory_path_for(results, "chain")
    assert direct != chain
    assert "direct" in direct.name and "chain" in chain.name


def test_each_model_gets_its_own_memory_file():
    """The 3B run is the control for the 7B run. It must not inherit its lessons."""
    a = memory_path_for(Path("data/bird/bird_results_3b.json"), "direct")
    b = memory_path_for(Path("data/bird/bird_results_7b.json"), "direct")
    assert a != b


def test_memory_files_sit_beside_their_results_file():
    results = Path("data/bird/bird_results_7b.json")
    assert memory_path_for(results, "direct").parent == results.parent


def test_two_arms_do_not_share_lessons(tmp_path):
    """The contamination this arm exists to avoid, exercised directly."""
    left = ErrorMemory(tmp_path / "left.json")
    right = ErrorMemory(tmp_path / "right.json")
    left.record(**LESSON)

    assert len(left) == 1
    assert len(right) == 0
    assert ErrorMemory(tmp_path / "right.json").recall(LESSON["question"]) == []


# ── the results file ─────────────────────────────────────────────────

def test_a_row_records_which_arm_it_belongs_to():
    row = Row(
        strategy="direct", question_id=1, db_id="x", difficulty="simple",
        correct=True, draft_correct=True, reason="match", sql="SELECT 1",
        calls=1, seconds=0.1, memory=True, lessons_recalled=2, lesson_learned=True,
    )
    assert row.memory is True
    assert row.lessons_recalled == 2


def test_results_files_from_before_phase_8_still_load():
    row = Row(
        strategy="direct", question_id=1, db_id="x", difficulty="simple",
        correct=True, draft_correct=True, reason="match", sql="SELECT 1",
        calls=1, seconds=0.1,
    )
    assert row.memory is False
    assert row.lessons_recalled == 0
    assert row.lesson_learned is False
