"""The Fixer agent — rewrites a query that failed.

The distinction that matters here is between blind retry and feedback-conditioned
repair. Asking the model to "try again" gives it the same inputs that produced the
wrong answer, and it will often produce the same wrong answer. The Fixer instead
receives *why* the previous attempt failed and is asked to address that
specifically.

Feedback comes from two sources, deliberately ranked:

  1. **The database.** `no such column: dept` is ground truth. It costs nothing,
     it is never wrong, and it is available before any model is consulted.
  2. **The Critic.** Catches what executes cleanly but answers the wrong
     question — the wrong aggregate, the wrong join.

Phase 1 measured why that order is not arbitrary: a 3B model rated a hallucinated
column as `schema_ok: true, confidence: 0.9`. Where a mechanical signal exists,
it leads.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..db.introspect import Schema
from ..db.safety import _strip_fences
from ..llm.client import LLMClient
from ..observability.trace import Trace
from .writer import _NullSpan

SYSTEM_PROMPT = """You repair broken {dialect} SELECT queries.

You will be given a question, the database schema, a query that failed, and the \
reason it failed. Rewrite the query so it answers the question and no longer has \
that problem.

Rules:
- Fix the specific problem described. Do not rewrite unrelated parts of the query.
- Use ONLY tables and columns from the schema. If told a column does not exist, \
use the suggested correct name.
- Return exactly one SELECT statement.

Return ONLY the corrected SQL. No explanation, no markdown fences."""

USER_PROMPT = """Question: {question}

Schema:
{schema}

This query failed:
{sql}

Why it failed:
{feedback}
{memory}
Write the corrected query."""


@dataclass
class Repair:
    """A repair attempt."""

    sql: str
    previous_sql: str
    feedback: str

    @property
    def changed(self) -> bool:
        return " ".join(self.sql.split()) != " ".join(self.previous_sql.split())


class Fixer:
    """Rewrites a failed query using the reason it failed."""

    name = "fixer"

    def __init__(self, client: LLMClient | None = None, schema: Schema | None = None):
        self.client = client or LLMClient(role="sql")
        self.schema = schema

    def fix(
        self,
        question: str,
        sql: str,
        feedback: str,
        *,
        schema: Schema | None = None,
        memory_context: str = "",
        trace: Trace | None = None,
        dialect: str | None = None,
    ) -> Repair:
        """Produce a corrected query."""
        from ..db.engine import dialect_name

        active_schema = schema or self.schema
        sql_dialect = dialect or dialect_name()

        span_ctx = (
            trace.span(self.name, "repairing query", reason=feedback[:120])
            if trace
            else _NullSpan()
        )
        with span_ctx as span:
            raw = self.client.chat(
                SYSTEM_PROMPT.format(dialect=sql_dialect),
                USER_PROMPT.format(
                    question=question,
                    schema=active_schema.render() if active_schema else "(unavailable)",
                    sql=sql,
                    feedback=feedback,
                    memory=f"\n{memory_context}" if memory_context else "",
                ),
            )
            fixed = _strip_fences(raw)
            repair = Repair(sql=fixed, previous_sql=sql, feedback=feedback)
            span.finish(sql=fixed, changed=repair.changed)

        return repair
