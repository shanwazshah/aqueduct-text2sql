"""The crew — what actually runs when you ask a question.

Phase 1 is deliberately the simplest possible crew: the Writer writes, the
Runner runs, done. No critics, no repair, no Lead deciding headcount. That comes
next, and it comes next *on purpose* — this version establishes the baseline
that every later addition has to beat.

If you cannot say how often a two-agent crew gets the right answer, you cannot
claim a seven-agent crew is worth its cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents.writer import Writer
from .config import settings
from .db.engine import QueryResult, run_query
from .db.introspect import Schema, load_schema
from .llm.client import LLMClient, Usage
from .observability.trace import Status, Trace


@dataclass
class Answer:
    """Everything one question produced.

    Carries the trace and usage alongside the result so the UI, the eval
    harness, and a human debugging a wrong answer all read from the same object.
    """

    question: str
    sql: str
    result: QueryResult
    trace: Trace
    usage: Usage
    explanation: str | None = None
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.result.ok

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
    """A minimal two-agent crew: Writer, then Runner."""

    def __init__(self, schema: Schema | None = None, usage: Usage | None = None):
        self.usage = usage or Usage()
        self.schema = schema or load_schema()
        self.writer = Writer(
            client=LLMClient(role="sql", usage=self.usage),
            schema=self.schema,
        )
        self._analyst = LLMClient(role="analyst", usage=self.usage)

    def ask(self, question: str, *, explain: bool = False) -> Answer:
        """Answer one question."""
        trace = Trace(question)

        draft = self.writer.write(question, trace=trace)

        with trace.span("runner", "executing query") as span:
            result = run_query(draft.sql)
            if result.ok:
                span.finish(rows=result.row_count, ms=round(result.elapsed_ms, 1))
            else:
                span.fail(result.error or "unknown error")

        explanation = None
        if explain and result.ok:
            with trace.span("analyst", "explaining result") as span:
                explanation = self._analyst.chat(
                    EXPLAIN_SYSTEM,
                    EXPLAIN_USER.format(
                        question=question, result=result.to_markdown(max_rows=15)
                    ),
                )
                span.finish(chars=len(explanation))

        trace.finish(Status.DONE if result.ok else Status.FAILED)

        return Answer(
            question=question,
            sql=result.sql if result.ok else draft.sql,
            result=result,
            trace=trace,
            usage=self.usage,
            explanation=explanation,
        )
