"""The crew — what actually runs when you ask a question.

Phase 1 was Writer plus Runner: write once, run once, accept whatever came back.
Phase 2 adds the ability to notice a bad answer and fix it.

The repair loop is deliberately configurable, because *which feedback signal does
the work* is an empirical question and the source notebooks assume an answer to
it. They lean on the model critiquing itself; Phase 1 measured a 3B model rating
a hallucinated column as `schema_ok: true, confidence: 0.9`. So `RepairMode`
exists to run the comparison rather than assume:

    NONE       Phase 1 baseline. Write once, run once.
    EXECUTION  Repair only on database errors. Free, ground truth, no extra call.
    CRITIQUE   Repair only on the Critic's opinion. What the notebooks do.
    BOTH       Execution first, then critique.

Every mode answers the same questions against the same grader, so the difference
between them is attributable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .agents.critic import Critic, Review
from .agents.fixer import Fixer
from .agents.memory import ErrorMemory
from .agents.writer import Writer
from .config import settings
from .db.engine import QueryResult, run_query
from .db.introspect import Schema, load_schema
from .llm.client import LLMClient, Usage
from .observability.trace import Status, Trace


class RepairMode(str, Enum):
    NONE = "none"
    EXECUTION = "execution"
    CRITIQUE = "critique"
    BOTH = "both"

    @property
    def uses_execution(self) -> bool:
        return self in (RepairMode.EXECUTION, RepairMode.BOTH)

    @property
    def uses_critique(self) -> bool:
        return self in (RepairMode.CRITIQUE, RepairMode.BOTH)


@dataclass
class Attempt:
    """One pass through write-or-fix, run, review."""

    sql: str
    result: QueryResult
    review: Review | None = None
    action: str = ""  # accepted | repairing | exhausted


@dataclass
class Answer:
    """Everything one question produced."""

    question: str
    sql: str
    result: QueryResult
    trace: Trace
    usage: Usage
    attempts: list[Attempt] = field(default_factory=list)
    explanation: str | None = None
    lesson_learned: bool = False

    @property
    def ok(self) -> bool:
        return self.result.ok

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    @property
    def was_repaired(self) -> bool:
        return len(self.attempts) > 1

    @property
    def agents_used(self) -> list[str]:
        return self.trace.agents_used

    def render(self) -> str:
        parts = [f"SQL:\n{self.sql}", ""]
        parts.append(self.result.to_markdown() if self.ok else f"FAILED: {self.result.error}")
        if self.explanation:
            parts += ["", self.explanation]
        return "\n".join(parts)


EXPLAIN_SYSTEM = """You explain query results to someone who did not write the SQL.

Answer the question directly in one or two sentences, using the numbers in the \
result. Do not describe the query, do not mention SQL, tables, or columns. If \
the result is empty, say plainly that no rows matched."""

EXPLAIN_USER = """Question: {question}

Result:
{result}

Answer the question."""


class Crew:
    """Writer, Runner, Critic, Fixer — with a configurable repair loop."""

    def __init__(
        self,
        *,
        strategy: "Strategy | str | None" = None,
        repair: RepairMode = RepairMode.BOTH,
        max_attempts: int | None = None,
        schema: Schema | None = None,
        usage: Usage | None = None,
        memory: ErrorMemory | None = None,
        use_memory: bool = True,
    ):
        from .strategies import DirectStrategy, get_strategy

        self.repair = repair
        self.max_attempts = max_attempts or settings.max_repair_attempts
        self.usage = usage or Usage()
        self.schema = schema or load_schema()
        self.use_memory = use_memory
        self.memory = memory if memory is not None else ErrorMemory()

        if strategy is None:
            self.strategy = DirectStrategy()
        elif isinstance(strategy, str):
            self.strategy = get_strategy(strategy)
        else:
            self.strategy = strategy

        self.writer = Writer(LLMClient(role="sql", usage=self.usage), self.schema)
        self.critic = Critic(LLMClient(role="critic", usage=self.usage), self.schema)
        self.fixer = Fixer(LLMClient(role="sql", usage=self.usage), self.schema)
        self._analyst = LLMClient(role="analyst", usage=self.usage)

    # ── main entry point ─────────────────────────────────────────────

    def ask(self, question: str, *, explain: bool = False) -> Answer:
        """Answer one question, repairing if the configured signals say to."""
        from .db.engine import dialect_name
        from .strategies import StrategyContext

        trace = Trace(question)
        memory_context = (
            self.memory.render_for_prompt(question) if self.use_memory else ""
        )

        # Generation is the strategy's job; execution and repair stay here, so a
        # difference between two strategies is attributable to generation alone.
        draft = self.strategy.generate(
            StrategyContext(
                question=question,
                schema=self.schema,
                trace=trace,
                usage=self.usage,
                dialect=dialect_name(),
                memory_context=memory_context,
            )
        )
        sql = draft.sql
        original_sql, original_error = sql, None

        attempts: list[Attempt] = []

        for attempt_no in range(1, self.max_attempts + 1):
            result = self._run(sql, trace)
            review = self._review(question, sql, result, trace)
            attempt = Attempt(sql=sql, result=result, review=review)

            feedback = self._feedback(result, review)
            last_attempt = attempt_no >= self.max_attempts

            if feedback is None:
                attempt.action = "accepted"
                attempts.append(attempt)
                break

            if last_attempt:
                attempt.action = "exhausted"
                attempts.append(attempt)
                break

            attempt.action = "repairing"
            attempts.append(attempt)

            if original_error is None:
                original_error = feedback

            repair = self.fixer.fix(
                question, sql, feedback,
                schema=self.schema, memory_context=memory_context, trace=trace,
            )

            # A Fixer that returns the same query has nothing more to offer;
            # looping again would just spend another 18 seconds to say so.
            if not repair.changed:
                attempt.action = "no-change"
                break

            sql = repair.sql

        final = attempts[-1]
        answer = Answer(
            question=question,
            sql=final.result.sql if final.result.ok else final.sql,
            result=final.result,
            trace=trace,
            usage=self.usage,
            attempts=attempts,
        )

        answer.lesson_learned = self._learn(
            question, original_sql, original_error, final
        )

        if explain and final.result.ok:
            answer.explanation = self._explain(question, final.result, trace)

        trace.finish(Status.DONE if final.result.ok else Status.FAILED)
        return answer

    # ── steps ────────────────────────────────────────────────────────

    def _run(self, sql: str, trace: Trace) -> QueryResult:
        with trace.span("runner", "executing query") as span:
            result = run_query(sql)
            if result.ok:
                span.finish(rows=result.row_count, ms=round(result.elapsed_ms, 1))
            else:
                span.fail(result.error or "unknown error")
        return result

    def _review(
        self, question: str, sql: str, result: QueryResult, trace: Trace
    ) -> Review | None:
        """Run the Critic, if this mode uses it.

        Skipped when the query already failed to execute: the database has given
        a precise reason, and spending a model call to obtain a vaguer one adds
        latency without adding information.
        """
        if not self.repair.uses_critique or not result.ok:
            return None
        return self.critic.review(question, sql, schema=self.schema, trace=trace)

    def _feedback(self, result: QueryResult, review: Review | None) -> str | None:
        """What to tell the Fixer, or None to accept the answer."""
        if self.repair is RepairMode.NONE:
            return None

        if not result.ok and self.repair.uses_execution:
            return f"The database rejected the query: {result.error}"

        if not result.ok:
            # Execution repair disabled (CRITIQUE mode): accept the failure so
            # the ablation measures critique alone rather than quietly using
            # the database error anyway.
            return None

        if review is not None and not review.is_clean:
            return review.feedback()

        return None

    def _learn(
        self,
        question: str,
        original_sql: str,
        original_error: str | None,
        final: Attempt,
    ) -> bool:
        """Record a lesson, but only for a repair that demonstrably worked.

        The bar is: the first attempt failed, the last one succeeded, and the SQL
        changed. Storing unverified corrections would put the crew's guesses into
        future prompts with the same authority as its knowledge.
        """
        if not self.use_memory or original_error is None or not final.result.ok:
            return False
        return (
            self.memory.record(question, original_sql, original_error, final.sql)
            is not None
        )

    def _explain(self, question: str, result: QueryResult, trace: Trace) -> str:
        with trace.span("analyst", "explaining result") as span:
            text = self._analyst.chat(
                EXPLAIN_SYSTEM,
                EXPLAIN_USER.format(
                    question=question, result=result.to_markdown(max_rows=15)
                ),
            )
            span.finish(chars=len(text))
        return text
