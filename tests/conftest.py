"""Shared test setup.

The demo database is generated, not committed - `data/db/*.db` is gitignored,
because a binary that `python -m aqueduct.cli seed` rebuilds in a second is not
source. But several test modules run real queries against it, so on a fresh
clone `pytest` failed with `no such table: employees` before this fixture
existed, and the fix ("run seed first") was written down nowhere.

Seeding here makes the suite self-contained.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def demo_database():
    """Build the demo database if it is missing or incomplete.

    Autouse, so it runs before any test touches the database whether or not the
    test asked for it. Cheap when the database is already there: one SELECT.
    """
    from aqueduct.db.engine import run_query
    from aqueduct.db.seed import seed

    if not run_query("SELECT 1 FROM employees").ok:
        seed()
