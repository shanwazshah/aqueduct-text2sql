"""Evaluator-optimiser — Workflow 4.

A generator writes, an evaluator scores against explicit criteria, and the score
plus its reasons feed back into the next attempt. Loop until it passes or the
budget runs out.

The difference from prompt chaining's verify stage — which the source notebook
raises and is worth being precise about — is that chaining verifies **once** as a
gate, while this **iterates**, and each attempt sees every criticism of the one
before. It is a loop with a score, not a checkpoint.

**A grader that cannot fail anything is not a grader.** Phase 2 measured a 3B
critic approving 22 of 22 queries, including ones that were wrong. A binary
pass/fail from that model would loop zero times and this strategy would collapse
into `direct` at twice the cost. So the evaluator scores each criterion
separately on a 0-3 scale: forcing a judgement per dimension makes "everything is
fine" a less available answer than a single overall verdict does.

Whether that actually works at 3B is an empirical question, and the point of
building it is to find out.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..agents.critic import check_against_schema
from .base import Draft, Strategy, StrategyContext, strip_sql
from .direct import SYSTEM_PROMPT, USER_PROMPT

# 0 = wrong, 1 = probably wrong, 2 = probably right, 3 = certainly right.
# A four-point scale has no comfortable middle, which is the point.
PASS_MARK = 3


class Evaluation(BaseModel):
    """A scored assessment of one attempt."""

    answers_question: int = Field(ge=0, le=3, description="Does it compute what was asked? 0-3")
    joins_correct: int = Field(ge=0, le=3, description="Are joins and their types right? 0-3")
    aggregation_correct: int = Field(ge=0, le=3, description="Aggregates and GROUP BY right? 0-3")
    filters_correct: int = Field(ge=0, le=3, description="Are WHERE/HAVING conditions right? 0-3")
    problems: list[str] = Field(default_factory=list, description="Specific, actionable problems.")

    @property
    def total(self) -> int:
        return (
            self.answers_question
            + self.joins_correct
            + self.aggregation_correct
            + self.filters_correct
        )

    @property
    def weakest(self) -> str:
        scores = {
            "answering the question": self.answers_question,
            "joins": self.joins_correct,
            "aggregation": self.aggregation_correct,
            "filters": self.filters_correct,
        }
        return min(scores, key=scores.get)

    def passes(self) -> bool:
        """Every criterion must be at the top of the scale.

        Requiring a perfect card is strict on purpose: a lenient bar plus a
        permissive grader means the loop never runs.
        """
        return all(
            score >= PASS_MARK
            for score in (
                self.answers_question,
                self.joins_correct,
                self.aggregation_correct,
                self.filters_correct,
            )
        )


EVALUATOR_BRIEF = """You grade SQL queries against the question they are meant to answer.

Score each criterion from 0 to 3:
  0 - definitely wrong
  1 - probably wrong
  2 - probably right
  3 - certainly right

Grade honestly. A query that computes something adjacent to what was asked is a 1 \
on answering the question, not a 2. A query using INNER JOIN where the question \
wants rows kept regardless is a 0 on joins.

Column names have already been verified - do not comment on them.

List specific problems. "The query counts orders but the question asks for their \
total value" is useful. "May be incorrect" is not."""


class EvaluatorOptimizerStrategy(Strategy):
    name = "eval_optimize"
    description = "Generate, grade against criteria, regenerate with the grades. Repeat."

    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds

    def generate(self, ctx: StrategyContext) -> Draft:
        sql = self._first_draft(ctx)
        history: list[str] = []
        rounds = 0
        best_sql, best_score = sql, -1

        for round_no in range(1, self.max_rounds + 1):
            evaluation = self._evaluate(ctx, sql, round_no)
            problems = list(evaluation.problems)

            # The mechanical check outranks the model's opinion; a hallucinated
            # column is a fact, not a judgement.
            schema_errors = check_against_schema(sql, ctx.schema)
            problems = schema_errors + problems

            score = evaluation.total - (10 * len(schema_errors))
            if score > best_score:
                best_sql, best_score = sql, score

            if evaluation.passes() and not schema_errors:
                break
            if round_no == self.max_rounds:
                break

            history.extend(problems)
            sql = self._revise(ctx, sql, problems, evaluation.weakest, history, round_no)
            rounds += 1

        # Return the best-scoring attempt, not merely the last one. Without this
        # a final revision that makes things worse would be what ships.
        final = sql if self._score_beats(ctx, sql, best_score) else best_sql
        return Draft(sql=final, notes={"rounds": rounds, "best_score": best_score})

    def _score_beats(self, ctx: StrategyContext, sql: str, best_score: int) -> bool:
        """Cheap tiebreak: prefer the last attempt only if it is schema-clean."""
        return not check_against_schema(sql, ctx.schema) and best_score < 0

    def _first_draft(self, ctx: StrategyContext) -> str:
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

    def _evaluate(self, ctx: StrategyContext, sql: str, round_no: int) -> Evaluation:
        with ctx.trace.span("evaluator", f"grading attempt {round_no}") as span:
            evaluation = ctx.client("critic").structured(
                EVALUATOR_BRIEF,
                f"Question: {ctx.question}\n\nSQL:\n{sql}\n\nSchema:\n{ctx.schema.render()}",
                Evaluation,
            )
            span.finish(
                score=f"{evaluation.total}/12",
                weakest=evaluation.weakest,
                problems=evaluation.problems,
            )
        return evaluation

    def _revise(
        self,
        ctx: StrategyContext,
        sql: str,
        problems: list[str],
        weakest: str,
        history: list[str],
        round_no: int,
    ) -> str:
        """Rewrite, with every criticism so far in view.

        Past rounds are included so the optimiser does not oscillate between two
        wrong queries, each fixing what the other broke.
        """
        current = "\n".join(f"  - {p}" for p in problems) or "  - does not fully answer the question"
        previous = ""
        if len(history) > len(problems):
            earlier = history[: -len(problems)] if problems else history
            previous = "\nAlready raised in earlier rounds:\n" + "\n".join(
                f"  - {p}" for p in earlier[-4:]
            )

        with ctx.trace.span("optimizer", f"revision {round_no}", focus=weakest) as span:
            raw = ctx.client("sql").chat(
                f"You improve {ctx.dialect} SELECT queries against reviewer feedback.\n\n"
                f"The weakest area of this query is: {weakest}. Address that first.\n"
                "Fix the listed problems without introducing new ones. "
                "Return ONLY the SQL.",
                f"Question: {ctx.question}\n\nCurrent SQL:\n{sql}\n\n"
                f"Problems:\n{current}{previous}\n\nSchema:\n{ctx.schema.render()}",
            )
            revised = strip_sql(raw)
            span.finish(sql=revised)
        return revised
