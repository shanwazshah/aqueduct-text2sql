"""The Critic agent — reviews SQL before anyone trusts it.

This agent exists to catch what execution cannot. A query with the wrong join,
the wrong aggregate, or a plausible-but-wrong column runs perfectly and returns a
confidently incorrect answer. The database has no opinion about whether the
answer is *right*; only about whether the SQL is *legal*.

That is the whole case for a critic — and the Phase 1 finding is the whole case
against relying on it. `qwen2.5-coder:3b` reviewed `SELECT dept FROM employees`,
having been told the column is `department`, and returned
`schema_ok: true, confidence: 0.9`.

So this agent is built with a hard rule: **checks that can be made mechanically
are made mechanically, and the model is only asked about what is left.** Column
existence is verified against the real schema in Python before the model is
consulted, because that is a set-membership test and a language model is the
wrong tool for it. What reaches the model is the genuinely semantic question:
does this query answer what was asked?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from pydantic import BaseModel, Field
from sqlglot import exp

from ..db.introspect import Schema
from ..llm.client import LLMClient, LLMError
from ..llm.types import UnitFloat
from ..observability.trace import Trace
from .writer import _NullSpan

SYSTEM_PROMPT = """You review SQL queries for correctness against a question.

You are NOT checking syntax or whether columns exist - that has already been \
verified mechanically. Judge only whether the query answers the question asked.

Look for:
- Wrong aggregate: COUNT where SUM was meant, AVG over the wrong column.
- Wrong join direction or type: an INNER JOIN that silently drops rows the \
question wants included.
- Missing or wrong filter: a condition the question implies but the query omits.
- Wrong grouping: results grouped at a different level than the question asks.
- Answering a different question than the one asked.

Be specific. "The query counts orders but the question asks for total value" is \
useful. "The query may be incorrect" is not.

If the query correctly answers the question, say so with high confidence and an \
empty issues list. Do not invent problems."""

USER_PROMPT = """Question: {question}

Schema:
{schema}

SQL:
{sql}
{execution}
Does this query correctly answer the question?"""


class Critique(BaseModel):
    """The Critic's structured verdict."""

    answers_question: bool = Field(description="Does the query answer what was asked?")
    confidence: UnitFloat = Field(default=0.5, description="Confidence from 0 to 1.")
    issues: list[str] = Field(default_factory=list, description="Specific problems found.")
    suggestion: str = Field(default="", description="One-line fix, or empty if none needed.")


@dataclass
class Review:
    """A critique combined with the mechanical checks that preceded it."""

    schema_errors: list[str] = field(default_factory=list)
    critique: Critique | None = None
    voted_bad: int = 0
    votes_cast: int = 0

    @property
    def has_schema_error(self) -> bool:
        return bool(self.schema_errors)

    @property
    def is_clean(self) -> bool:
        if self.schema_errors:
            return False
        if self.critique is None:
            return True
        return self.critique.answers_question and not self.critique.issues

    def feedback(self) -> str:
        """Render as repair instructions for the Fixer."""
        parts = []
        if self.schema_errors:
            parts.append(
                "Schema errors (verified against the real database):\n"
                + "\n".join(f"  - {e}" for e in self.schema_errors)
            )
        if self.critique and self.critique.issues:
            parts.append(
                "Review found:\n"
                + "\n".join(f"  - {i}" for i in self.critique.issues)
            )
        if self.critique and self.critique.suggestion:
            parts.append(f"Suggested fix: {self.critique.suggestion}")
        return "\n\n".join(parts)


# ── mechanical checks ────────────────────────────────────────────────


def check_against_schema(sql: str, schema: Schema, dialect: str = "sqlite") -> list[str]:
    """Verify that every table and column in a query actually exists.

    This is a set-membership test with an exact answer. Handing it to a language
    model — as the source notebooks do via `schema_ok` in the critique JSON —
    turns a decidable question into a probabilistic one, and Phase 1 measured a
    3B model getting it backwards with 0.9 confidence.

    **Parsed, not pattern-matched.** The first implementation used regular
    expressions over the query text and reported three classes of column that
    were never columns at all:

        SELECT SUM(amount) AS total_value FROM orders
            -> "column 'total_value' does not exist"      (an output alias)

        SELECT CAST(SUM(CASE WHEN status = 'cancelled' ...) AS REAL) ...
            -> "column 'cancelled' does not exist"        (a string literal)
            -> "column 'real' does not exist"             (a type name)
            -> "column 'cast' does not exist"             (a function)

    Every one of those fired on a query that was correct, and each would have
    triggered a pointless repair — and, through the router, spent a critic call
    to review a query that had nothing wrong with it. A keyword list cannot fix
    this: the problem is that identifiers only have meaning in a grammatical
    position, and only a parser knows the position.

    Returns human-readable errors, phrased for the Fixer to act on. Anything the
    parser cannot resolve is passed over in silence — a false negative costs a
    missed hint, while a false positive costs a wasted rewrite of correct SQL.
    """
    if not sql or not sql.strip():
        return []

    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return []  # unparseable SQL is the safety guard's problem, not ours
    if tree is None:
        return []

    columns_by_table = {
        t.name.lower(): {c.name.lower() for c in t.columns} for t in schema.tables
    }
    errors: list[str] = []

    # Names introduced by the query itself are not schema objects: CTEs and
    # subquery aliases are legitimate targets that no table list will contain.
    local_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE) if cte.alias}
    for subquery in tree.find_all(exp.Subquery):
        if subquery.alias:
            local_names.add(subquery.alias.lower())

    # ── tables ──
    referenced: set[str] = set()
    aliases: dict[str, str] = {}

    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if not name or name in local_names:
            continue
        if name in columns_by_table:
            referenced.add(name)
            aliases[name] = name
            if table.alias:
                aliases[table.alias.lower()] = name
        else:
            close = _closest(name, columns_by_table.keys())
            hint = f" Did you mean '{close}'?" if close else ""
            message = f"table '{table.name}' does not exist.{hint}"
            if message not in errors:
                errors.append(message)

    # A query built on a CTE cannot have its columns resolved against the base
    # schema, so column checking is skipped rather than guessed at.
    if local_names:
        return errors

    # ── columns ──
    # Output aliases are defined by the query, so `... AS total_value` must not
    # be looked up as though it were a stored column.
    defined = {
        alias.alias.lower() for alias in tree.find_all(exp.Alias) if alias.alias
    }

    in_scope: set[str] = set()
    for name in referenced:
        in_scope |= columns_by_table[name]

    for column in tree.find_all(exp.Column):
        name = (column.name or "").lower()
        qualifier = (column.table or "").lower()
        if not name or name == "*" or name in defined:
            continue

        if qualifier:
            table = aliases.get(qualifier)
            if table is None:
                continue  # unknown qualifier: already reported, or out of scope
            if name not in columns_by_table[table]:
                close = _closest(name, columns_by_table[table])
                hint = f" Did you mean '{close}'?" if close else ""
                message = f"column '{column.name}' does not exist on table '{table}'.{hint}"
                if message not in errors:
                    errors.append(message)
        elif in_scope and name not in in_scope:
            # Unqualified. Suggestions come from the single referenced table
            # when there is one; searching the whole database produced
            # `dept -> budget` when the answer was plainly `department_id`.
            candidates = (
                columns_by_table[next(iter(referenced))] if len(referenced) == 1 else in_scope
            )
            close = _closest(name, candidates)
            hint = f" Did you mean '{close}'?" if close else ""
            message = f"column '{column.name}' does not exist.{hint}"
            if message not in errors:
                errors.append(message)

    return errors


def _closest(name: str, candidates) -> str | None:
    """Nearest known identifier, for a 'did you mean' hint.

    The hint matters more than it looks: `no such column: dept` tells the Fixer
    something is wrong, while `did you mean 'department'` tells it what to write.
    """
    import difflib

    matches = difflib.get_close_matches(name, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


# ── the agent ────────────────────────────────────────────────────────


class Critic:
    """Reviews a query. Mechanical checks first, model second."""

    name = "critic"

    def __init__(self, client: LLMClient | None = None, schema: Schema | None = None):
        self.client = client or LLMClient(role="critic")
        self.schema = schema

    def review(
        self,
        question: str,
        sql: str,
        *,
        schema: Schema | None = None,
        execution_error: str | None = None,
        ask_model: bool = True,
        trace: Trace | None = None,
        label: str = "reviewing SQL",
    ) -> Review:
        """Review a query and return findings."""
        active_schema = schema or self.schema
        review = Review()

        span_ctx = trace.span(self.name, label) if trace else _NullSpan()
        with span_ctx as span:
            if active_schema is not None:
                review.schema_errors = check_against_schema(sql, active_schema)

            # If the schema is provably wrong there is nothing to deliberate
            # about, and asking the model would only spend 18 seconds to agree.
            if review.schema_errors:
                span.finish(verdict="schema-error", issues=review.schema_errors)
                return review

            if not ask_model:
                span.finish(verdict="mechanical-only")
                return review

            execution_note = ""
            if execution_error:
                execution_note = f"\nThe database rejected this query: {execution_error}\n"

            try:
                review.critique = self.client.structured(
                    SYSTEM_PROMPT,
                    USER_PROMPT.format(
                        question=question,
                        schema=active_schema.render() if active_schema else "(unavailable)",
                        sql=sql,
                        execution=execution_note,
                    ),
                    Critique,
                )
            except LLMError as e:
                # A critic that cannot answer must not block the pipeline; the
                # query still faces execution, which is the stronger check.
                span.fail(str(e))
                return review

            span.finish(
                verdict="ok" if review.is_clean else "issues",
                confidence=review.critique.confidence,
                issues=review.critique.issues,
            )

        return review
