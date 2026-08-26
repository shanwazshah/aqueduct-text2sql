"""The router — Workflow 2, redirected by the evidence.

**What this was going to be.** Route each question to a generation strategy:
cheap questions to `direct`, hard ones to `orchestrator`. That is the obvious
reading of routing, and it is what the project was designed around.

**Why it is not that.** Phase 3 measured every strategy's generation quality
separately from what the repair layer rescued:

    direct / parallel / eval_optimize   90.9%
    chain                               50.0%
    orchestrator                        40.9%
    react                                4.5%

There is nothing to route *to*. No strategy generates better than one call, and
the elaborate ones generate far worse. A router choosing between them could only
lose. So generation is fixed at `direct` and the router is pointed at the
decision the data says is live.

**What it actually decides: how much verification a question deserves.**

Phase 2 measured where verification value comes from:

  * Execution repair: **+4.5 points**, and it costs a call only when a query
    actually fails. Free on success.
  * Critic review: **+0.0 points**, and it costs a call on *every* question.

So execution repair is always on — there is no argument for disabling something
that is free when it succeeds and correct when it fires. The only real decision
left is whether a given question is worth spending a critic call on, and the
router's job is to spend that call where it might pay.

**The router is mechanical, not a model.** This follows the rule the project has
applied since Phase 2: decidable questions are decided in code. Query complexity
is readable directly from the parsed SQL — joins, aggregates, subqueries, GROUP
BY — at zero cost and with no chance of hallucination. Asking a 3B model to rate
complexity would spend the very call the router exists to avoid, which would be
self-defeating. An LLM router is implemented alongside for comparison, because
"the free signal is as good as the paid one" is a claim that should be measured
rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import sqlglot
from sqlglot import exp

from .crew import RepairMode
from .db.introspect import Schema
from .llm.client import LLMClient, LLMError
from .observability.trace import Trace


class Tier(str, Enum):
    """How much checking a question gets."""

    TRUST = "trust"      # execution repair only — the free signal
    VERIFY = "verify"    # add a critic review before accepting

    @property
    def repair_mode(self) -> RepairMode:
        return RepairMode.EXECUTION if self is Tier.TRUST else RepairMode.BOTH


@dataclass
class Verdict:
    """A routing decision, with its reasons."""

    tier: Tier
    score: int
    signals: list[str] = field(default_factory=list)
    source: str = "mechanical"

    def explain(self) -> str:
        if not self.signals:
            return f"{self.tier.value} (nothing risky found)"
        return f"{self.tier.value} (score {self.score}: {', '.join(self.signals)})"


# ── risk signals, weighted ───────────────────────────────────────────
#
# Weights come from Phase 3's contested set: the questions where strategies
# disagreed were per-group extremes, anti-joins, multi-hop joins, self-joins and
# integer division. Those shapes are visible in the SQL itself, so they are what
# the scorer looks for.

VERIFY_THRESHOLD = 3


def score_sql(sql: str, dialect: str = "sqlite") -> tuple[int, list[str]]:
    """Rate a query's risk from its structure. No model, no cost.

    Returns a score and the signals that produced it, so a routing decision can
    always be explained — a router that cannot say why it spent money is not
    worth having.
    """
    if not sql or not sql.strip():
        # Nothing to inspect. Something upstream failed; check it.
        return 99, ["no query produced"]

    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return 99, ["query does not parse"]

    if tree is None:
        return 99, ["query does not parse"]

    # sqlglot does not raise on prose. `parse_one("this is not sql")` returns a
    # column expression quite happily, so a missing `except` is not enough — the
    # root has to be checked. Without this, garbage scored 0 and was *trusted*,
    # which is the worst possible direction for the error to run in.
    if not isinstance(tree, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.Subquery)):
        return 99, ["not a SELECT query"]

    score = 0
    signals: list[str] = []

    joins = list(tree.find_all(exp.Join))
    if joins:
        # One join is routine. Three tables chained together is where the
        # multi-hop failures in Phase 3 lived.
        score += len(joins)
        signals.append(f"{len(joins)} join{'s' if len(joins) > 1 else ''}")

    # sqlglot leaves `side` unset for an INNER JOIN and "LEFT"/"RIGHT" for an
    # outer one, so plain truthiness is the whole test.
    if any(j.args.get("side") for j in joins):
        score += 1
        signals.append("outer join")

    # A subquery means the model is reasoning about two result sets at once —
    # per-group extremes and anti-joins both land here.
    subqueries = [
        node for node in tree.find_all(exp.Select) if node is not tree
    ]
    if subqueries:
        score += 2 * len(subqueries)
        signals.append(f"{len(subqueries)} subquery" + ("s" if len(subqueries) > 1 else ""))

    if list(tree.find_all(exp.Exists)):
        score += 1
        signals.append("EXISTS")

    if tree.args.get("group"):
        score += 1
        signals.append("GROUP BY")

    if tree.args.get("having"):
        score += 2
        signals.append("HAVING")

    if list(tree.find_all(exp.Window)):
        score += 2
        signals.append("window function")

    aggregates = [
        node
        for node in tree.find_all(exp.AggFunc)
    ]
    if aggregates:
        score += 1
        signals.append(f"{len(aggregates)} aggregate" + ("s" if len(aggregates) > 1 else ""))

    # Arithmetic between columns is where amount-vs-price and integer division
    # went wrong.
    if list(tree.find_all(exp.Div)):
        score += 2
        signals.append("division")
    if list(tree.find_all(exp.Mul)):
        score += 1
        signals.append("multiplication")

    if list(tree.find_all(exp.Case)):
        score += 1
        signals.append("CASE")

    # A self-join is the same table joined to itself *in one scope*. Counting
    # every table reference in the tree instead labels
    # `... WHERE salary > (SELECT AVG(salary) FROM employees)` a self-join,
    # because `employees` appears on both sides of the subquery boundary. The
    # routing decision came out the same, but the stated reason was wrong — and
    # a router that cannot explain itself accurately is not worth having.
    scope_tables = [
        t.name.lower()
        for t in tree.find_all(exp.Table)
        if t.name and t.parent_select is tree
    ]
    if len(scope_tables) != len(set(scope_tables)):
        score += 2
        signals.append("self-join")

    return score, signals


class Router:
    """Decides the verification tier for each question."""

    name = "lead"

    def __init__(self, threshold: int = VERIFY_THRESHOLD, schema: Schema | None = None):
        self.threshold = threshold
        self.schema = schema

    def route(
        self,
        question: str,
        sql: str,
        *,
        dialect: str = "sqlite",
        trace: Trace | None = None,
    ) -> Verdict:
        """Decide how much checking this query gets."""
        score, signals = score_sql(sql, dialect)

        # A query referencing something that does not exist is going to fail at
        # execution anyway, and that failure is a better signal than a critic.
        if self.schema is not None:
            from .agents.critic import check_against_schema

            if check_against_schema(sql, self.schema):
                score += 3
                signals.append("schema mismatch")

        tier = Tier.VERIFY if score >= self.threshold else Tier.TRUST
        verdict = Verdict(tier=tier, score=score, signals=signals)

        if trace is not None:
            with trace.span(self.name, "sizing the check") as span:
                span.finish(tier=tier.value, score=score, signals=signals)

        return verdict


# ── the LLM comparison ───────────────────────────────────────────────


class LLMRouter:
    """Asks a model to rate complexity instead of parsing the query.

    Built only to test whether the free signal is as good as the paid one. It
    spends exactly the call the mechanical router exists to save, so for it to be
    worth using it would have to route *better*, not merely as well.
    """

    name = "lead"

    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient(role="lead")

    def route(
        self,
        question: str,
        sql: str,
        *,
        dialect: str = "sqlite",
        trace: Trace | None = None,
    ) -> Verdict:
        from pydantic import BaseModel, Field

        class Rating(BaseModel):
            needs_review: bool = Field(
                description="True if this query is complex or risky enough to warrant review."
            )
            reason: str = Field(default="", description="One short sentence.")

        try:
            rating = self.client.structured(
                "You decide whether a SQL query needs a careful review before its "
                "answer is trusted.\n\n"
                "Say it needs review when the query involves multiple joins, "
                "subqueries, per-group extremes, anti-joins, division, or "
                "conditional aggregation - anywhere a subtle mistake would still "
                "return plausible-looking rows.\n\n"
                "Say it does not when the query is a straightforward lookup, count, "
                "or single-table filter.",
                f"Question: {question}\n\nSQL:\n{sql}",
                Rating,
            )
            tier = Tier.VERIFY if rating.needs_review else Tier.TRUST
            verdict = Verdict(
                tier=tier, score=int(rating.needs_review),
                signals=[rating.reason] if rating.reason else [], source="llm",
            )
        except LLMError as e:
            # A router that cannot decide should check rather than trust.
            verdict = Verdict(Tier.VERIFY, 0, [f"router failed: {e}"], source="llm")

        if trace is not None:
            with trace.span(self.name, "sizing the check") as span:
                span.finish(tier=verdict.tier.value, signals=verdict.signals, source="llm")

        return verdict
