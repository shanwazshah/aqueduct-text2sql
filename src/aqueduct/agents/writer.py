"""The Writer agent — turns a question into SQL.

This is the first agent in the crew and the one whose output everything else
reacts to. It does exactly one thing: read a question and a schema, and emit a
query. It does not execute, validate, or explain — the Runner, the Critics, and
the Analyst each own their own step, and keeping those separate is what lets the
Lead spin up only the ones a given question needs.

The prompt is deliberately blunt about the failure modes that matter for small
models, because a 3B model needs the traps named rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import settings
from ..db.introspect import Schema, load_schema
from ..db.safety import _strip_fences
from ..llm.client import LLMClient
from ..observability.trace import Trace

SYSTEM_PROMPT = """You are a SQL specialist. You write a single {dialect} SELECT \
query that answers the user's question.

Rules:
- Use ONLY tables and columns that appear in the schema given to you. Never invent one.
- Write exactly one SELECT statement. No INSERT, UPDATE, DELETE, DROP or DDL.
- Qualify every column with its table when more than one table is involved.
- Match string values to the sample values shown in the schema, exactly as spelled.
- Use LEFT JOIN when a row should still appear even if the joined side is missing.
- When you aggregate, every non-aggregated selected column must be in GROUP BY.

Return ONLY the SQL query. No explanation, no markdown fences, no commentary."""

USER_PROMPT = """Database schema:
{schema}

Question: {question}
{extra}
Write the {dialect} SELECT query that answers this question."""


@dataclass
class Draft:
    """A written query, before anyone has checked it."""

    sql: str
    question: str
    tables_seen: list[str]


class Writer:
    """Writes SQL. Does not run it."""

    name = "writer"

    def __init__(self, client: LLMClient | None = None, schema: Schema | None = None):
        self.client = client or LLMClient(role="sql")
        self._schema = schema

    @property
    def schema(self) -> Schema:
        if self._schema is None:
            self._schema = load_schema()
        return self._schema

    def write(
        self,
        question: str,
        *,
        tables: list[str] | None = None,
        feedback: str | None = None,
        trace: Trace | None = None,
        dialect: str | None = None,
    ) -> Draft:
        """Produce a query for `question`.

        `tables` narrows the schema — the Scout agent will supply it later so a
        wide database does not flood the prompt. `feedback` carries a previous
        failure, which is how the Fixer reuses this same agent rather than
        duplicating the prompt.
        """
        from ..db.engine import dialect_name

        sql_dialect = dialect or dialect_name()
        schema = self.schema.subset(tables) if tables else self.schema

        extra = ""
        if feedback:
            # Put the correction after the question, so it is the last thing
            # the model reads before writing.
            extra = (
                f"\nA previous attempt failed. Do not repeat this mistake:\n"
                f"{feedback}\n"
            )

        system = SYSTEM_PROMPT.format(dialect=sql_dialect)
        user = USER_PROMPT.format(
            schema=schema.render(),
            question=question,
            extra=extra,
            dialect=sql_dialect,
        )

        span_ctx = (
            trace.span(self.name, "writing SQL", tables=schema.table_names)
            if trace
            else _NullSpan()
        )
        with span_ctx as span:
            raw = self.client.chat(system, user)
            sql = _strip_fences(raw)
            if hasattr(span, "finish"):
                span.finish(sql=sql, model=self.client.model)

        return Draft(sql=sql, question=question, tables_seen=schema.table_names)


class _NullSpan:
    """Lets tracing be optional without scattering `if trace:` through the code."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def finish(self, *_, **__):
        return self
