"""Tests for the BIRD sweep's results-file format.

A results file outlives the code that wrote it. Kaggle sessions produce them,
they get downloaded, and they are re-read months later by a re-grade or a
report - so adding a field must never make the existing ones unloadable.
"""

from __future__ import annotations

import dataclasses

from aqueduct.eval.bird_run import Row

PRE_PHASE_7 = {
    "strategy": "direct",
    "question_id": 12,
    "db_id": "california_schools",
    "difficulty": "simple",
    "correct": True,
    "draft_correct": True,
    "reason": "match",
    "sql": "SELECT COUNT(*) FROM schools",
    "calls": 1,
    "seconds": 4.2,
    "repaired": False,
    "agents": ["writer", "runner"],
}


def test_a_results_file_written_before_phase_7_still_loads():
    """The Kaggle runs already on disk predate `draft_sql`."""
    row = Row(**PRE_PHASE_7)
    assert row.draft_sql == ""
    assert row.correct is True


def test_the_draft_query_is_stored_not_just_its_grade():
    """Without the query itself, a grader change cannot be applied to a finished
    sweep, and the gen EX column - the one every claim rests on - costs a full
    GPU re-run to recover."""
    fields = {f.name for f in dataclasses.fields(Row)}
    assert "draft_sql" in fields
    assert "draft_correct" in fields
