"""Parallelisation — Workflow 3.

Write the query once, then have several reviewers examine it simultaneously, each
looking for a different class of mistake, and repair only if enough of them
object.

Two things the pattern is claimed to buy:

  * **Coverage.** A reviewer asked to check one thing checks it more carefully
    than a reviewer asked to check everything.
  * **Robustness.** Requiring a quorum means one reviewer's bad call does not
    trigger a pointless rewrite. Phase 2 found a general-purpose 3B critic
    hallucinating approval; a vote is one way to damp that.

**On the word "parallel".** The calls are dispatched concurrently, and against a
batching server like vLLM that is a genuine latency win. Against a single Ollama
instance on one GPU, requests queue — so locally this pattern costs three times
the tokens and roughly three times the wall clock. That is worth measuring rather
than assuming, and it is the kind of difference the two-tier setup exists to
expose.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..agents.critic import check_against_schema
from .base import Draft, Strategy, StrategyContext, strip_sql
from .direct import SYSTEM_PROMPT, USER_PROMPT


class ReviewVote(BaseModel):
    """One reviewer's verdict.

    Approval is a vote, not a score. A confidence number here would be
    decorative - the quorum counts objections - and bounded numbers from a model
    need defending (see llm/types.py).
    """

    approved: bool = Field(description="True if the query is correct in this reviewer's area.")
    issue: str = Field(default="", description="The single most important problem, or empty.")


@dataclass(frozen=True)
class Reviewer:
    """A reviewer with one job."""

    name: str
    focus: str
    brief: str


REVIEWERS = (
    Reviewer(
        name="join-reviewer",
        focus="joins",
        brief=(
            "You review ONLY the join logic of a SQL query.\n\n"
            "Check: are the join conditions on the correct key columns? Is INNER "
            "JOIN used where LEFT JOIN is needed, silently dropping rows the "
            "question wants kept? Would any join produce a cartesian product? Is a "
            "table joined that the question does not need?\n\n"
            "Ignore aggregation, filters and formatting - other reviewers cover those."
        ),
    ),
    Reviewer(
        name="aggregation-reviewer",
        focus="aggregation",
        brief=(
            "You review ONLY the aggregation logic of a SQL query.\n\n"
            "Check: is the aggregate function the one the question asked for - "
            "COUNT where SUM was meant, or the wrong column inside AVG? Does GROUP "
            "BY contain every non-aggregated selected column? Is the grouping at "
            "the level the question asked for? Should a condition be in HAVING "
            "rather than WHERE?\n\n"
            "Ignore joins and filters - other reviewers cover those."
        ),
    ),
    Reviewer(
        name="intent-reviewer",
        focus="intent",
        brief=(
            "You review ONLY whether a SQL query answers the question that was asked.\n\n"
            "Check: does it compute the thing requested, or something adjacent to "
            "it? Are there filters in the query the question never asked for? Are "
            "there conditions the question stated that the query ignores? Does it "
            "return the right granularity - one row where a breakdown was wanted, "
            "or the reverse?\n\n"
            "Ignore syntax and column names - those are already verified."
        ),
    ),
)


class ParallelStrategy(Strategy):
    name = "parallel"
    description = "One draft, several specialised reviewers voting, repair on quorum."

    def __init__(self, votes_to_repair: int = 2, max_repairs: int = 1):
        self.votes_to_repair = votes_to_repair
        self.max_repairs = max_repairs

    def generate(self, ctx: StrategyContext) -> Draft:
        sql = self._draft(ctx)
        rounds = 0

        for _ in range(self.max_repairs):
            objections = self._review(ctx, sql)
            if len(objections) < self.votes_to_repair:
                break
            sql = self._repair(ctx, sql, objections)
            rounds += 1

        return Draft(sql=sql, notes={"repair_rounds": rounds})

    def _draft(self, ctx: StrategyContext) -> str:
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
        return sql

    def _review(self, ctx: StrategyContext, sql: str) -> list[str]:
        """Dispatch every reviewer at once and collect the objections.

        The mechanical schema check runs first and counts as its own objection.
        It is free, deterministic, and it means the model reviewers never have to
        be asked about column existence — the question Phase 2 showed they answer
        badly.
        """
        objections: list[str] = []

        schema_errors = check_against_schema(sql, ctx.schema)
        if schema_errors:
            with ctx.trace.span("schema-check", "verifying columns") as span:
                span.finish(errors=schema_errors)
            # A hallucinated column is decisive on its own. No vote required.
            return schema_errors

        with ctx.trace.span("reviewers", f"{len(REVIEWERS)} reviewers in parallel") as span:
            with ThreadPoolExecutor(max_workers=len(REVIEWERS)) as pool:
                votes = list(pool.map(lambda r: self._one_review(ctx, r, sql), REVIEWERS))

            for reviewer, vote in zip(REVIEWERS, votes):
                if vote is not None and not vote.approved and vote.issue:
                    objections.append(f"[{reviewer.focus}] {vote.issue}")

            span.finish(
                approved=sum(1 for v in votes if v is not None and v.approved),
                objections=objections,
            )

        return objections

    def _one_review(self, ctx: StrategyContext, reviewer: Reviewer, sql: str) -> ReviewVote | None:
        """Run one reviewer. Failures return None rather than blocking the quorum."""
        from ..llm.client import LLMError

        try:
            return ctx.client("critic").structured(
                reviewer.brief,
                f"Question: {ctx.question}\n\nSQL:\n{sql}\n\nSchema:\n{ctx.schema.render()}",
                ReviewVote,
            )
        except LLMError:
            return None

    def _repair(self, ctx: StrategyContext, sql: str, objections: list[str]) -> str:
        issues = "\n".join(f"  - {o}" for o in objections)
        with ctx.trace.span("fixer", "repairing query", objections=len(objections)) as span:
            raw = ctx.client("sql").chat(
                f"You repair {ctx.dialect} SELECT queries. Reviewers have raised the "
                "problems below. Fix them and change nothing else. Return ONLY the SQL.",
                f"Question: {ctx.question}\n\nSQL:\n{sql}\n\n"
                f"Problems raised:\n{issues}\n\nSchema:\n{ctx.schema.render()}",
            )
            fixed = strip_sql(raw)
            span.finish(sql=fixed)
        return fixed
