"""Direct generation — one call, one query.

The control group. Every other strategy spends more calls than this one, so this
is what they have to beat to justify themselves.

It is not a strawman. A good schema card plus a prompt that names the specific
failure modes gets a long way, and Phase 1 measured this at 90.9% on the demo set
before any repair was added. Strategies that cost seven calls per question should
be held to that standard, not to a deliberately weak baseline.
"""

from __future__ import annotations

from .base import Draft, Strategy, StrategyContext, strip_sql

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
{memory}
Write the {dialect} SELECT query that answers this question."""


class DirectStrategy(Strategy):
    name = "direct"
    description = "Single LLM call. The baseline every other strategy must beat."

    def generate(self, ctx: StrategyContext) -> Draft:
        with ctx.trace.span("writer", "writing SQL") as span:
            raw = ctx.client("sql").chat(
                SYSTEM_PROMPT.format(dialect=ctx.dialect),
                USER_PROMPT.format(
                    schema=ctx.schema.render(),
                    question=ctx.question,
                    memory=f"\n{ctx.memory_context}" if ctx.memory_context else "",
                    dialect=ctx.dialect,
                ),
            )
            sql = strip_sql(raw)
            span.finish(sql=sql)

        return Draft(sql=sql, notes={"calls": 1})
