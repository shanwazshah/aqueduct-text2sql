"""Database access — the only place in the project that touches a real database.

Everything goes through `run_query`, which guards the SQL first and converts
failures into structured results rather than exceptions. That matters because a
database error is not a crash here: it is feedback the Fixer agent learns from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from time import perf_counter

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ..config import settings
from .safety import guard


@dataclass
class QueryResult:
    """Outcome of one query attempt.

    `error` is deliberately a plain sentence rather than a raw traceback — it is
    fed straight back into a repair prompt, so it needs to read like advice.
    """

    ok: bool
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        """Zero rows. Valid SQL, but usually a sign the filters are wrong."""
        return self.ok and not self.rows

    def to_markdown(self, max_rows: int = 20) -> str:
        """Render as a markdown table for prompts and the UI."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if not self.rows:
            return "(no rows)"

        shown = self.rows[:max_rows]
        head = "| " + " | ".join(self.columns) + " |"
        rule = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join("NULL" if v is None else str(v) for v in row) + " |"
            for row in shown
        ]
        table = "\n".join([head, rule, *body])
        if len(self.rows) > max_rows:
            table += f"\n\n_({len(self.rows) - max_rows} more rows not shown)_"
        return table


@lru_cache(maxsize=8)
def get_engine(db_url: str | None = None) -> Engine:
    """Build a SQLAlchemy engine. Cached so we reuse one pool per URL."""
    url = db_url or settings.db_url
    kwargs: dict = {"future": True}
    if url.startswith("sqlite"):
        # Streamlit and the agent crew both touch the DB from several threads.
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def dialect_name(db_url: str | None = None) -> str:
    """Which SQL dialect to parse and generate for."""
    return get_engine(db_url).dialect.name


def run_query(
    sql: str,
    db_url: str | None = None,
    max_rows: int | None = None,
) -> QueryResult:
    """Guard, execute, and package a query.

    Never raises for bad SQL. A rejected or failing query comes back as a
    QueryResult with `ok=False` and a readable reason, because the crew's repair
    loop is driven by exactly that string.
    """
    engine = get_engine(db_url)
    cap = max_rows or settings.max_rows

    verdict = guard(sql, dialect=engine.dialect.name, max_rows=cap)
    if not verdict.ok:
        return QueryResult(ok=False, sql=sql, error=verdict.reason)

    safe_sql = verdict.sql
    start = perf_counter()
    try:
        with engine.connect() as conn:
            cursor = conn.execute(text(safe_sql))
            columns = list(cursor.keys())
            rows = [tuple(r) for r in cursor.fetchall()]
    except Exception as e:
        return QueryResult(
            ok=False,
            sql=safe_sql,
            error=_readable_error(e),
            elapsed_ms=(perf_counter() - start) * 1000,
        )

    return QueryResult(
        ok=True,
        sql=safe_sql,
        columns=columns,
        rows=rows,
        elapsed_ms=(perf_counter() - start) * 1000,
    )


def _readable_error(e: Exception) -> str:
    """Trim a driver exception down to the part a model can act on.

    SQLAlchemy wraps driver errors with the full statement and parameter dump
    appended. Feeding all of that back to a small model buries the actual
    problem, so we keep only the first line.
    """
    message = str(getattr(e, "orig", e)).strip()
    return message.splitlines()[0] if message else type(e).__name__
