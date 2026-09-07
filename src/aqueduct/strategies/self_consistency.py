"""Self-consistency — the control the deep agent has to beat.

A deep agent spends roughly ten calls where `direct` spends one. So `direct` at
one call is not its baseline. Its baseline is **whatever else ten calls can
buy**, and the obvious other thing to buy is ten samples of the single-call
prompt with the answers voted on.

This is the control that almost nobody runs, and it frequently wins. It is also
exactly the shape of control that overturned this project's own headline in
Phase 6: hold the interesting variable fixed, vary the boring one, and find out
how much of the effect was never about the interesting variable at all.

**Voting is on the result set, not the SQL.** Two correct queries can differ in
every character and agree on every row, which is the whole reason this project
grades by execution. Grouping by query text would count `COUNT(*)` and
`COUNT(id)` as disagreeing and hand the vote to whichever spelling happened to
repeat.

**A query that fails to execute does not get a vote.** It has not produced an
answer to agree with. If every sample fails, the first one is returned so the
repair layer still has something to work on - and the metadata says the vote was
empty, so a score cannot be read as consensus when there was none.

**Temperature is above zero, deliberately.** Sampling at 0 returns the same
answer n times and the cache would serve nine of them for free, which would look
like a bargain and measure nothing. The response cache stores nothing above 0,
so these are genuinely n calls and the cost in the leaderboard is real.
"""

from __future__ import annotations

from collections import defaultdict

from ..db.engine import run_query
from .base import Draft, Strategy, StrategyContext, strip_sql
from .direct import SYSTEM_PROMPT, USER_PROMPT

DEFAULT_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7


class SelfConsistencyStrategy(Strategy):
    """Sample the single-call prompt n times, run each, vote on the result set."""

    name = "self_consistency"
    description = "n samples of the baseline prompt, voting on the answer they produce."

    def __init__(
        self,
        samples: int = DEFAULT_SAMPLES,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.samples = samples
        self.temperature = temperature

    def generate(self, ctx: StrategyContext) -> Draft:
        client = ctx.client("sql", temperature=self.temperature)
        system = SYSTEM_PROMPT.format(dialect=ctx.dialect)
        user = USER_PROMPT.format(
            schema=ctx.schema.render(),
            question=ctx.question,
            memory=f"\n{ctx.memory_context}" if ctx.memory_context else "",
            dialect=ctx.dialect,
        )

        drafts: list[str] = []
        votes: dict[tuple, list[str]] = defaultdict(list)

        with ctx.trace.span("self-consistency", f"{self.samples} samples") as span:
            for i in range(self.samples):
                try:
                    sql = strip_sql(client.chat(system, user))
                except Exception as e:  # one bad sample must not lose the question
                    with ctx.trace.span("sample", f"sample {i + 1}") as s:
                        s.fail(str(e))
                    continue
                if not sql:
                    continue
                drafts.append(sql)

                result = run_query(sql, db_url=ctx.db_url)
                with ctx.trace.span("sample", f"sample {i + 1}") as s:
                    s.finish(sql=sql, rows=result.row_count if result.ok else None)

                if result.ok:
                    votes[_answer_key(result)].append(sql)

            if votes:
                winner = max(votes.values(), key=len)
                sql, agreement = winner[0], len(winner)
            else:
                # Nothing executed. Hand back the first draft rather than an
                # empty string, and record that no vote took place.
                sql, agreement = (drafts[0] if drafts else ""), 0

            span.finish(
                samples=len(drafts),
                distinct_answers=len(votes),
                agreement=agreement,
                sql=sql,
            )

        return Draft(
            sql=sql,
            notes={
                "samples": len(drafts),
                "distinct_answers": len(votes),
                "agreement": agreement,
                "unanimous": bool(votes) and len(votes) == 1,
            },
        )


def _answer_key(result) -> tuple:
    """A hashable form of a result set, for grouping equal answers.

    Rows are compared as an unordered multiset of stringified tuples. This is
    deliberately looser than the grader: the grader decides whether an answer is
    *right*, while this only needs to decide whether two samples said the *same
    thing*, and being strict about ordering here would split a genuine consensus
    across several buckets.
    """
    return tuple(sorted(str(tuple(row)) for row in result.rows))
