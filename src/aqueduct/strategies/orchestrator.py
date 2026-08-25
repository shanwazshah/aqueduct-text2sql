"""Orchestrator-worker — Workflow 5.

An orchestrator reads the question, decides which specialists the job needs, runs
them, and synthesises their findings into one query.

This is the most expensive strategy in the project — up to seven calls for a
question `direct` answers in one — and it is the one the router exists to avoid
using unnecessarily. Establishing what it is worth is the point of building it.

**The orchestrator picks the crew.** That is the part worth preserving from the
pattern, and the part most implementations quietly drop by running every worker
every time. A single-table count needs a schema worker and nothing else; a
four-table revenue breakdown needs joins, aggregation and filters. Sizing the
crew to the question here is a rehearsal for what the Lead agent will do across
whole strategies in Phase 4.

Workers with no dependency on each other are dispatched concurrently. As with
Workflow 3, that is a real latency win on a batching server and a queue on one
local GPU.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..llm.client import LLMError
from .base import Draft, Strategy, StrategyContext, strip_sql


class Plan(BaseModel):
    """The orchestrator's staffing decision."""

    complexity: str = Field(description="simple, moderate, or complex")
    workers: list[str] = Field(
        description="Which specialists are needed: schema, join, aggregation, filter, subquery"
    )
    reasoning: str = Field(default="", description="One sentence on why.")


class WorkerFinding(BaseModel):
    """One specialist's contribution."""

    findings: list[str] = Field(description="Concrete, specific statements for the synthesiser.")
    sql_fragment: str = Field(default="", description="The clause this worker is responsible for, if any.")


@dataclass(frozen=True)
class Worker:
    key: str
    label: str
    brief: str


WORKERS = {
    w.key: w
    for w in (
        Worker(
            key="schema",
            label="schema worker",
            brief=(
                "You identify exactly which tables and columns a question needs.\n\n"
                "List each column as table.column, using names exactly as they appear "
                "in the schema. Include columns needed only for joining. Do not write "
                "a full query."
            ),
        ),
        Worker(
            key="join",
            label="join worker",
            brief=(
                "You plan the joins for a query.\n\n"
                "State each join as an explicit condition (table_a.col = table_b.col), "
                "following foreign keys shown in the schema. Say whether each should be "
                "INNER or LEFT, and why - if the question wants rows kept even when the "
                "other side is missing, it is LEFT. Do not write a full query."
            ),
        ),
        Worker(
            key="aggregation",
            label="aggregation worker",
            brief=(
                "You plan the aggregation for a query.\n\n"
                "State which aggregate functions are needed and over which columns, what "
                "GROUP BY must contain, and whether any condition belongs in HAVING "
                "rather than WHERE. If the question implies a ranking, say what to ORDER "
                "BY and whether a LIMIT is needed. Do not write a full query."
            ),
        ),
        Worker(
            key="filter",
            label="filter worker",
            brief=(
                "You extract filter conditions from a question.\n\n"
                "State each WHERE condition explicitly. Match string values to the sample "
                "values in the schema, exactly as spelled. Include ONLY conditions the "
                "question actually states - do not invent plausible ones. If the question "
                "implies no filter, say so. Do not write a full query."
            ),
        ),
        Worker(
            key="subquery",
            label="subquery worker",
            brief=(
                "You plan subqueries and nested logic.\n\n"
                "Questions needing a per-group extreme, a comparison against an overall "
                "aggregate, or an anti-join ('never', 'no', 'without') usually need a "
                "correlated subquery, NOT EXISTS, or a HAVING clause against a scalar "
                "subquery. State the structure precisely. Do not write a full query."
            ),
        ),
    )
}


class OrchestratorStrategy(Strategy):
    name = "orchestrator"
    description = "An orchestrator staffs specialists to the question, then synthesises."

    def generate(self, ctx: StrategyContext) -> Draft:
        plan = self._plan(ctx)
        findings = self._run_workers(ctx, plan.workers)
        sql = self._synthesise(ctx, plan, findings)
        return Draft(
            sql=sql,
            notes={
                "complexity": plan.complexity,
                "workers": list(findings),
                "worker_count": len(findings),
            },
        )

    # ── plan ─────────────────────────────────────────────────────────

    def _plan(self, ctx: StrategyContext) -> Plan:
        with ctx.trace.span("orchestrator", "planning the work") as span:
            try:
                plan = ctx.client("lead").structured(
                    "You decide which specialists are needed to build a SQL query.\n\n"
                    "Available specialists:\n"
                    "  schema      - which tables and columns (almost always needed)\n"
                    "  join        - only if more than one table is involved\n"
                    "  aggregation - only if counting, summing, averaging, or grouping\n"
                    "  filter      - only if the question states conditions\n"
                    "  subquery    - only for per-group extremes, comparisons against an\n"
                    "                overall aggregate, or 'never/no/without' questions\n\n"
                    "Assign ONLY what this question needs. A simple count over one table "
                    "needs schema alone. Over-staffing wastes work and invites the "
                    "synthesiser to include clauses nobody asked for.",
                    f"Question: {ctx.question}\n\nSchema:\n{ctx.schema.render_compact()}",
                    Plan,
                )
            except LLMError:
                plan = Plan(complexity="unknown", workers=["schema"], reasoning="planning failed")

            # The schema worker is the floor: without it the synthesiser has no
            # grounded column list and starts inventing.
            chosen = [w for w in plan.workers if w in WORKERS]
            if "schema" not in chosen:
                chosen.insert(0, "schema")
            plan.workers = chosen

            span.finish(complexity=plan.complexity, workers=chosen, why=plan.reasoning)
        return plan

    # ── workers ──────────────────────────────────────────────────────

    def _run_workers(self, ctx: StrategyContext, keys: list[str]) -> dict[str, WorkerFinding]:
        """Run the chosen specialists concurrently."""
        if not keys:
            return {}

        with ctx.trace.span("workers", f"{len(keys)} specialists") as span:
            with ThreadPoolExecutor(max_workers=len(keys)) as pool:
                results = list(pool.map(lambda k: (k, self._run_one(ctx, k)), keys))

            findings = {k: v for k, v in results if v is not None}
            span.finish(ran=list(findings), failed=[k for k, v in results if v is None])

        return findings

    def _run_one(self, ctx: StrategyContext, key: str) -> WorkerFinding | None:
        worker = WORKERS[key]
        try:
            return ctx.client("critic").structured(
                worker.brief,
                f"Question: {ctx.question}\n\nSchema:\n{ctx.schema.render()}",
                WorkerFinding,
            )
        except LLMError:
            return None  # a missing specialist degrades the plan, not the run

    # ── synthesis ────────────────────────────────────────────────────

    def _synthesise(
        self, ctx: StrategyContext, plan: Plan, findings: dict[str, WorkerFinding]
    ) -> str:
        report = []
        for key, finding in findings.items():
            lines = "\n".join(f"    - {f}" for f in finding.findings)
            block = f"  {WORKERS[key].label}:\n{lines}"
            if finding.sql_fragment:
                block += f"\n    clause: {finding.sql_fragment}"
            report.append(block)

        with ctx.trace.span("synthesiser", "assembling the query") as span:
            raw = ctx.client("sql").chat(
                f"You assemble one {ctx.dialect} SELECT query from specialist reports.\n\n"
                "Use the tables and columns the schema worker identified, the joins the "
                "join worker specified with the types it gave, and the filters the filter "
                "worker listed - no others.\n\n"
                "Where two reports conflict, prefer the more conservative reading. Where a "
                "report is silent, leave that clause out rather than inventing it.\n\n"
                "Return ONLY the SQL. No explanation, no fences.",
                f"Question: {ctx.question}\n\n"
                f"Specialist reports:\n" + "\n\n".join(report) + "\n\n"
                f"Schema:\n{ctx.schema.render()}",
            )
            sql = strip_sql(raw)
            span.finish(sql=sql)
        return sql
