"""Routing comparison — does deciding beat always doing the same thing?

Four configurations over the same questions, all using `direct` generation
(Phase 3: nothing generates better) and all with execution repair always on
(Phase 2: +4.5 points, free on success):

    trust-all      never spend a critic call
    verify-all     always spend a critic call
    router         spend it when the parsed SQL looks risky   (free to decide)
    llm-router     spend it when a model says so              (costs a call)

For the router to be worth anything it has to land at `verify-all`'s accuracy
for closer to `trust-all`'s cost. If `trust-all` and `verify-all` score the same
— which Phase 2 suggests they will, since the critic bought nothing there — then
the honest finding is that there is nothing to route, and the router's value is
whatever it saves rather than what it gains.

    python -m aqueduct.eval.routing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from ..crew import Crew, RepairMode
from ..db.introspect import load_schema
from ..router import LLMRouter, Router, Tier
from .demo_set import ALL_QUESTIONS
from .metrics import execution_accuracy


@dataclass
class Config:
    label: str
    repair: RepairMode
    router: object | None = None


@dataclass
class Outcome:
    label: str
    correct: int = 0
    total: int = 0
    calls: int = 0
    seconds: float = 0.0
    verified: int = 0            # questions that got a critic review
    decisions: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)

    @property
    def ex(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    @property
    def calls_per_question(self) -> float:
        return self.calls / self.total if self.total else 0.0


def evaluate(config: Config, schema) -> Outcome:
    outcome = Outcome(label=config.label)

    for question in ALL_QUESTIONS:
        crew = Crew(
            strategy="direct",
            repair=config.repair,
            router=config.router,
            schema=schema,
            use_memory=False,
        )
        start = perf_counter()
        answer = crew.ask(question.question)
        grade = execution_accuracy(answer.sql, question.gold_sql)

        outcome.total += 1
        outcome.correct += int(grade.correct)
        outcome.calls += answer.calls
        outcome.seconds += perf_counter() - start

        routed = getattr(answer, "routed", None)
        if routed is not None:
            if routed.tier is Tier.VERIFY:
                outcome.verified += 1
            outcome.decisions.append(f"{question.id}: {routed.explain()}")
        elif config.repair.uses_critique:
            outcome.verified += 1

        if not grade.correct:
            outcome.missed.append(question.id)

    return outcome


def main() -> None:
    schema = load_schema()

    configs = [
        Config("trust-all", RepairMode.EXECUTION),
        Config("verify-all", RepairMode.BOTH),
        Config("router", RepairMode.EXECUTION, Router(schema=schema)),
        Config("llm-router", RepairMode.EXECUTION, LLMRouter()),
    ]

    outcomes = []
    for config in configs:
        print(f"\n{'=' * 60}\n  {config.label}\n{'=' * 60}")
        outcome = evaluate(config, schema)
        outcomes.append(outcome)
        print(
            f"  EX {outcome.ex:.1f}%  ({outcome.correct}/{outcome.total})   "
            f"verified {outcome.verified}/{outcome.total}   "
            f"calls/q {outcome.calls_per_question:.2f}   {outcome.seconds:.0f}s"
        )

    print(f"\n\n{'=' * 74}")
    print("ROUTING COMPARISON".center(74))
    print("=" * 74)
    print(f"{'config':<14}{'EX':>8}{'correct':>10}{'verified':>11}{'calls/q':>10}{'time':>10}")
    print("-" * 74)
    for outcome in outcomes:
        print(
            f"{outcome.label:<14}{outcome.ex:>7.1f}%"
            f"{outcome.correct:>7}/{outcome.total:<2}"
            f"{outcome.verified:>8}/{outcome.total:<2}"
            f"{outcome.calls_per_question:>10.2f}"
            f"{outcome.seconds:>9.0f}s"
        )
    print("=" * 74)

    for outcome in outcomes:
        if outcome.missed:
            print(f"\n{outcome.label} missed: {', '.join(outcome.missed)}")

    router_outcome = next((o for o in outcomes if o.label == "router"), None)
    if router_outcome and router_outcome.decisions:
        print("\nrouter decisions:")
        for decision in router_outcome.decisions:
            print(f"  {decision}")


if __name__ == "__main__":
    main()
