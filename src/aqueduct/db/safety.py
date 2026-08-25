"""The safety guard.

The notebooks' approach was to write "NEVER run DELETE, DROP, UPDATE, INSERT"
into the persona and hope the model complies. That is a request, not a control —
and a 3B model will eventually ignore it.

This module makes destructive SQL structurally impossible instead. Every query
the crew produces is parsed into an AST and inspected before it reaches the
database. A rule the model cannot violate beats a rule it is asked to respect.

Checks applied, in order:
  1. It must parse at all.
  2. Exactly one statement (blocks `SELECT 1; DROP TABLE users`).
  3. The top-level node must be a read (SELECT / UNION / WITH...SELECT).
  4. No write or admin node anywhere in the tree, at any depth.
  5. A row cap is injected or tightened.
"""

from __future__ import annotations

from dataclasses import dataclass

import sqlglot
from sqlglot import exp


def _optional(*names: str) -> tuple[type, ...]:
    """Collect expression classes that exist in this sqlglot version.

    sqlglot renames nodes between releases; looking them up by name keeps the
    guard working across versions instead of crashing on an AttributeError.
    """
    found = []
    for name in names:
        node = getattr(exp, name, None)
        if isinstance(node, type):
            found.append(node)
    return tuple(found)


# Anything that writes, destroys, or reconfigures. `Command` is sqlglot's
# catch-all for statements it does not model in detail — PRAGMA, ATTACH,
# VACUUM — and none of those belong in a read-only analytics path.
FORBIDDEN = _optional(
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "TruncateTable",
    "Grant", "Revoke", "Command", "Set", "Use", "Transaction", "Commit",
    "Rollback", "AlterTable", "Attach", "Detach", "Copy", "Merge",
)

# Legal shapes for a top-level read.
ALLOWED_ROOTS = _optional("Select", "Union", "Except", "Intersect", "Subquery", "With")

# Functions that can touch the filesystem or load code, even from a SELECT.
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}


@dataclass(frozen=True)
class SafetyVerdict:
    """Result of guarding a query."""

    ok: bool
    sql: str
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok


def guard(sql: str, dialect: str = "sqlite", max_rows: int = 500) -> SafetyVerdict:
    """Validate and normalise a generated query.

    Returns a verdict carrying either the safe, row-capped SQL or the reason it
    was rejected. The reason is written to be useful to the Fixer agent, so it
    doubles as repair feedback.
    """
    if not sql or not sql.strip():
        return SafetyVerdict(False, sql, "Query is empty.")

    cleaned = _strip_fences(sql)

    try:
        statements = [s for s in sqlglot.parse(cleaned, dialect=dialect) if s is not None]
    except Exception as e:  # sqlglot raises several parse error types
        return SafetyVerdict(False, cleaned, f"Query is not valid {dialect} SQL: {e}")

    if not statements:
        return SafetyVerdict(False, cleaned, "No SQL statement found.")

    if len(statements) > 1:
        return SafetyVerdict(
            False, cleaned,
            f"Expected one statement, found {len(statements)}. "
            "Chained statements are not permitted.",
        )

    tree = statements[0]

    if not isinstance(tree, ALLOWED_ROOTS):
        return SafetyVerdict(
            False, cleaned,
            f"Only read queries are allowed; this is a {type(tree).__name__.upper()} statement.",
        )

    # Depth-first sweep — a write can hide inside a CTE or subquery.
    for node in tree.walk():
        if isinstance(node, FORBIDDEN):
            return SafetyVerdict(
                False, cleaned,
                f"Query contains a forbidden {type(node).__name__.upper()} operation.",
            )
        if isinstance(node, exp.Anonymous):
            fn = (node.name or "").lower()
            if fn in FORBIDDEN_FUNCTIONS:
                return SafetyVerdict(False, cleaned, f"Function '{fn}' is not permitted.")

    capped = _apply_row_cap(tree, max_rows)

    # Comments are dropped rather than re-emitted. sqlglot rewrites `--` line
    # comments into `/* */` blocks, and while it does correctly escape a `*/`
    # inside the payload, relying on that is a thin margin. Comments carry no
    # value for us, so removing them removes the question entirely.
    emitted = capped.sql(dialect=dialect, pretty=True, comments=False)

    # Verify our own output. Everything above reasons about the tree we parsed;
    # this confirms the string we actually hand to the driver still parses to
    # exactly one read statement. Cheap, and it closes the gap between what we
    # validated and what we execute.
    if not _is_single_read(emitted, dialect):
        return SafetyVerdict(False, cleaned, "Query failed post-generation safety re-check.")

    return SafetyVerdict(True, emitted, None)


def _is_single_read(sql: str, dialect: str) -> bool:
    """Re-parse emitted SQL and confirm it is still one read statement."""
    try:
        statements = [s for s in sqlglot.parse(sql, dialect=dialect) if s is not None]
    except Exception:
        return False
    if len(statements) != 1 or not isinstance(statements[0], ALLOWED_ROOTS):
        return False
    return not any(isinstance(node, FORBIDDEN) for node in statements[0].walk())


def _apply_row_cap(tree: exp.Expression, max_rows: int) -> exp.Expression:
    """Add a LIMIT, or tighten one that is too generous.

    A model that forgets LIMIT on a million-row table should not be able to
    stall the app, and an agent asking for 100,000 rows is not going to read them.
    """
    if not isinstance(tree, exp.Select):
        # Set operations (UNION etc.) get wrapped so the cap applies to the whole result.
        return exp.select("*").from_(tree.subquery("capped")).limit(max_rows)

    existing = tree.args.get("limit")
    if existing is None:
        return tree.limit(max_rows)

    try:
        current = int(existing.expression.name)
    except (AttributeError, ValueError, TypeError):
        return tree.limit(max_rows)  # non-literal limit — replace it

    return tree.limit(max_rows) if current > max_rows else tree


def _strip_fences(sql: str) -> str:
    """Remove markdown code fences a model may have wrapped the SQL in.

    Structured decoding removes most of this problem, but SQL is returned as
    plain text rather than JSON, so the habit survives and is cheap to undo.
    """
    text = sql.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()
