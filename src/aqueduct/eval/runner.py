"""Run a question set through a crew and grade it.

Kept deliberately plain at this stage. It gets a resume file and concurrency
when the Kaggle sweeps start; right now the job is to produce one honest number
for the two-agent baseline so later phases have something to beat.
"""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Iterable

from ..crew import Answer
from .demo_set import ALL_QUESTIONS, Question
from .metrics import Grade, Report, execution_accuracy


def evaluate(
    ask: Callable[[str], Answer],
    questions: Iterable[Question] | None = None,
    *,
    verbose: bool = True,
    on_result: Callable[[Question, Answer, Grade], None] | None = None,
) -> tuple[Report, list[tuple[Question, Answer, Grade]]]:
    """Ask every question, grade every answer.

    `ask` is any callable taking a question string and returning an Answer, so
    this works unchanged against every strategy added later.
    """
    questions = list(questions or ALL_QUESTIONS)
    report = Report()
    rows: list[tuple[Question, Answer, Grade]] = []
    start = perf_counter()

    for i, question in enumerate(questions, 1):
        if verbose:
            print(f"[{i:>2}/{len(questions)}] {question.id}  {question.question[:60]}")

        answer = ask(question.question)
        grade = execution_accuracy(answer.sql, question.gold_sql)

        report.add(grade, question.difficulty, question.trap)
        report.llm_calls += answer.calls
        rows.append((question, answer, grade))

        if verbose:
            mark = "PASS" if grade.correct else "FAIL"
            print(f"          {mark}  {grade.reason}")
            if not grade.correct:
                print(f"          sql: {_one_line(answer.sql)}")

        if on_result:
            on_result(question, answer, grade)

    report.seconds = perf_counter() - start
    return report, rows


def failures(rows: list[tuple[Question, Answer, Grade]]) -> str:
    """Render just the failures, with the gold query alongside.

    This is the output worth reading — it is where the next thing to build
    becomes obvious.
    """
    lines = []
    for question, answer, grade in rows:
        if grade.correct:
            continue
        lines += [
            f"--- {question.id} [{question.difficulty}]"
            + (f" trap={question.trap}" if question.trap else ""),
            f"  Q:    {question.question}",
            f"  why:  {grade.reason}",
            f"  got:  {_one_line(answer.sql)}",
            f"  gold: {_one_line(question.gold_sql)}",
            "",
        ]
    return "\n".join(lines) if lines else "(no failures)"


def _one_line(sql: str) -> str:
    return " ".join(sql.split())


if __name__ == "__main__":
    from ..crew import Crew

    crew = Crew()
    report, rows = evaluate(lambda q: crew.ask(q))
    print("\n" + "=" * 60)
    print(report.render())
    print("=" * 60 + "\n")
    print(failures(rows))
