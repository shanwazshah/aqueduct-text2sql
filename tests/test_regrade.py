"""Tests for the offline re-grader.

The re-grader applies a grader change to work already finished. If it is wrong,
it silently rewrites results that took hours of GPU to produce - so it is tested
against the demo database, which conftest seeds.
"""

from __future__ import annotations

import json

import pytest

from aqueduct.eval.regrade import regrade


def write_rows(tmp_path, rows: list[dict]):
    path = tmp_path / "results.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_regrade_reproduces_grades_that_were_already_right(tmp_path):
    path = write_rows(tmp_path, [
        {
            "strategy": "direct", "question_id": "q01", "correct": True,
            "reason": "match", "sql": "SELECT COUNT(*) FROM employees",
            "calls": 1, "seconds": 0.0, "agents": [], "repaired": False,
            "draft_sql": "SELECT COUNT(*) FROM employees", "draft_correct": True,
        },
    ])
    tallies, _ = regrade(path)
    assert tallies["direct"].changes == []
    assert tallies["direct"].final_now == 1
    assert tallies["direct"].gen_now == 1


def test_regrade_catches_a_grade_that_was_wrong(tmp_path):
    """A stored `correct: true` on a query that does not match must be caught."""
    path = write_rows(tmp_path, [
        {
            "strategy": "direct", "question_id": "q01", "correct": True,
            "reason": "match", "sql": "SELECT COUNT(*) FROM departments",
            "calls": 1, "seconds": 0.0, "agents": [], "repaired": False,
            "draft_sql": "SELECT COUNT(*) FROM departments", "draft_correct": True,
        },
    ])
    tallies, updated = regrade(path)
    moved = tallies["direct"].changes
    assert {c.column for c in moved} == {"final", "gen"}
    assert all(c.direction == "pass->FAIL" for c in moved)
    assert updated[0]["correct"] is False


def test_regrade_reports_missing_drafts_rather_than_assuming_them(tmp_path):
    """A row with no `draft_sql` is not evidence, and must not be counted as if it were."""
    path = write_rows(tmp_path, [
        {
            "strategy": "direct", "question_id": "q01", "correct": True,
            "reason": "match", "sql": "SELECT COUNT(*) FROM employees",
            "calls": 1, "seconds": 0.0, "agents": [], "repaired": False,
            "draft_correct": True,
        },
    ])
    tallies, _ = regrade(path)
    assert tallies["direct"].gen_ungradable == 1
    assert tallies["direct"].gen_now == 0


def test_regrade_does_not_modify_the_file(tmp_path):
    rows = [{
        "strategy": "direct", "question_id": "q01", "correct": True, "reason": "match",
        "sql": "SELECT COUNT(*) FROM departments", "calls": 1, "seconds": 0.0,
        "agents": [], "repaired": False,
        "draft_sql": "SELECT COUNT(*) FROM departments", "draft_correct": True,
    }]
    path = write_rows(tmp_path, rows)
    original = path.read_text(encoding="utf-8")
    regrade(path)
    assert path.read_text(encoding="utf-8") == original


def test_regrade_skips_rows_with_no_reference_question(tmp_path):
    path = write_rows(tmp_path, [{
        "strategy": "direct", "question_id": "not-a-question", "correct": True,
        "reason": "match", "sql": "SELECT 1", "calls": 1, "seconds": 0.0,
        "agents": [], "repaired": False, "draft_sql": "SELECT 1", "draft_correct": True,
    }])
    tallies, _ = regrade(path)
    assert tallies["direct"].skipped == 1
    assert tallies["direct"].rows == 0


def test_bird_results_without_databases_are_refused(tmp_path):
    """A BIRD row needs its database. Guessing would grade against the wrong schema."""
    path = write_rows(tmp_path, [{
        "strategy": "direct", "question_id": 1, "db_id": "california_schools",
        "difficulty": "simple", "correct": True, "draft_correct": True,
        "reason": "match", "sql": "SELECT 1", "calls": 1, "seconds": 0.0,
    }])
    with pytest.raises(SystemExit, match="--databases"):
        regrade(path)
