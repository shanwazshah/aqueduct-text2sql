"""Prompt chaining — Workflow 1.

Decompose SQL generation into stages, each doing one thing, each receiving only
what the previous stage produced:

    intent -> ground -> generate -> verify -> repair

The idea, from Anthropic's "Building Effective Agents": a model asked to do one
narrow thing does it better than a model asked to do five things at once, and the
gate between stages catches errors before they compound.

Two departures from the source notebook, both deliberate:

**Schema grounding is done in Python.** The notebook asks a model to map extracted
entities onto real tables by feeding it the whole schema as JSON and asking for
the relevant subset. That is a retrieval problem with an exact answer, and Phase 2
established that handing exact problems to a 3B model is how hallucinated columns
get rated `schema_ok: true`. Here the model proposes table names and the code
resolves them against the real schema, keeping only what exists.

**The verify stage judges semantics only.** Column existence is already guaranteed
by grounding, so asking about it again would spend a call to re-derive something
known.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..db.introspect import Schema
from .base import Draft, Strategy, StrategyContext, strip_sql


class Intent(BaseModel):
    """Stage 1 output — what the question is asking for."""

    operation: str = Field(description="count, list, aggregate, compare, rank, or filter")
    entities: list[str] = Field(default_factory=list, description="Things mentioned, e.g. employees, orders")
    filters: list[str] = Field(default_factory=list, description="Conditions, e.g. status is shipped")
    aggregations: list[str] = Field(default_factory=list, description="e.g. average salary, total revenue")
    grouping: list[str] = Field(default_factory=list, description="What to group by, if anything")
    ordering: str = Field(default="", description="Sort requirement, or empty")


class TableSelection(BaseModel):
    """Stage 2 output — which tables the query needs."""

    tables: list[str] = Field(description="Exact table names needed to answer the question")
    reasoning: str = Field(default="", description="One sentence on why these tables")


class Verification(BaseModel):
    """Stage 4 output — does the query answer the question?

    No confidence field. Nothing read it, and a decorative bounded number is a
    liability: the first run died because the model returned 100 for a field
    documented as 0.0-1.0. A field that drives no decision should not exist.
    """

    passes: bool
    issues: list[str] = Field(default_factory=list)


class ChainStrategy(Strategy):
    name = "chain"
    description = "Five stages: intent, grounding, generation, verification, repair."

    def __init__(self, max_repairs: int = 1):
        self.max_repairs = max_repairs

    def generate(self, ctx: StrategyContext) -> Draft:
        intent = self._extract_intent(ctx)
        tables = self._ground(ctx, intent)
        narrowed = ctx.schema.subset(tables) if tables else ctx.schema

        sql = self._write(ctx, intent, narrowed)

        for _ in range(self.max_repairs):
            verdict = self._verify(ctx, sql, narrowed)
            if verdict.passes and not verdict.issues:
                break
            sql = self._repair(ctx, sql, verdict, narrowed)

        return Draft(sql=sql, notes={"tables": tables, "operation": intent.operation})

    # ── stage 1: what is being asked ─────────────────────────────────

    def _extract_intent(self, ctx: StrategyContext) -> Intent:
        with ctx.trace.span("intent", "parsing the question") as span:
            intent = ctx.client("critic").structured(
                "You analyse questions asked of a SQL database. Break the question "
                "into its parts. Do not write SQL. Name only what the question "
                "actually says - do not invent filters or groupings.",
                f"Question: {ctx.question}",
                Intent,
            )
            span.finish(operation=intent.operation, entities=intent.entities)
        return intent

    # ── stage 2: which tables ────────────────────────────────────────

    def _ground(self, ctx: StrategyContext, intent: Intent) -> list[str]:
        """Pick the tables needed, then resolve them against the real schema.

        The model's answer is treated as a proposal. Names that do not exist are
        dropped rather than passed downstream, so a hallucinated table cannot
        reach the generation stage.
        """
        with ctx.trace.span("scout", "finding relevant tables") as span:
            selection = ctx.client("critic").structured(
                "You map a question onto the tables of a database.\n\n"
                "Return the exact names of every table needed to answer it, and no "
                "others. Include tables needed only to join two others together. "
                "Use names exactly as they appear in the schema.",
                f"Question: {ctx.question}\n\n"
                f"Intent: {intent.model_dump_json()}\n\n"
                f"Schema:\n{ctx.schema.render_compact()}",
                TableSelection,
            )

            known = {t.name.lower(): t.name for t in ctx.schema.tables}
            resolved = [known[t.lower()] for t in selection.tables if t.lower() in known]
            dropped = [t for t in selection.tables if t.lower() not in known]

            # Selecting nothing usable is worse than selecting everything; fall
            # back to the full schema rather than generating against a void.
            if not resolved:
                resolved = ctx.schema.table_names
                span.finish(tables=resolved, fallback=True, dropped=dropped)
            else:
                span.finish(tables=resolved, dropped=dropped)

        return resolved

    # ── stage 3: write it ────────────────────────────────────────────

    def _write(self, ctx: StrategyContext, intent: Intent, schema: Schema) -> str:
        with ctx.trace.span("writer", "writing SQL") as span:
            raw = ctx.client("sql").chat(
                f"You write a single {ctx.dialect} SELECT query.\n\n"
                "Use ONLY the tables and columns given. Qualify columns with their "
                "table. Every non-aggregated selected column must appear in GROUP BY.\n\n"
                "Return ONLY the SQL. No explanation, no fences.",
                f"Question: {ctx.question}\n\n"
                f"What the question asks for:\n{intent.model_dump_json(indent=2)}\n\n"
                f"Schema (use only these):\n{schema.render()}\n"
                f"{ctx.memory_context}",
            )
            sql = strip_sql(raw)
            span.finish(sql=sql)
        return sql

    # ── stage 4: the gate ────────────────────────────────────────────

    def _verify(self, ctx: StrategyContext, sql: str, schema: Schema) -> Verification:
        with ctx.trace.span("verifier", "checking the query") as span:
            verdict = ctx.client("critic").structured(
                "You check whether a SQL query answers the question asked.\n\n"
                "Column existence has already been verified - do not comment on it. "
                "Judge only: does this compute what the question asked for? Check "
                "the aggregate, the join type, the filters, and the grouping level.\n\n"
                "If it is correct, pass it. Do not invent problems.",
                f"Question: {ctx.question}\n\nSQL:\n{sql}\n\nSchema:\n{schema.render()}",
                Verification,
            )
            span.finish(passes=verdict.passes, issues=verdict.issues)
        return verdict

    # ── stage 5: conditional repair ──────────────────────────────────

    def _repair(
        self, ctx: StrategyContext, sql: str, verdict: Verification, schema: Schema
    ) -> str:
        issues = "\n".join(f"  - {i}" for i in verdict.issues) or "  - does not answer the question"
        with ctx.trace.span("fixer", "repairing query") as span:
            raw = ctx.client("sql").chat(
                f"You repair {ctx.dialect} SELECT queries. Fix the listed problems "
                "and change nothing else. Return ONLY the corrected SQL.",
                f"Question: {ctx.question}\n\nSQL:\n{sql}\n\n"
                f"Problems:\n{issues}\n\nSchema:\n{schema.render()}",
            )
            fixed = strip_sql(raw)
            span.finish(sql=fixed)
        return fixed
