"""The deep agent — Phase 9.

Every decomposed strategy in this project lost to a single call, and the
mechanism was the same each time: **each stage inherits the previous stage's
errors and has no way to detect them.** The orchestrator's synthesiser follows
its specialists faithfully, wrong join key included.

The one thing that has earned its keep in seven phases is execution feedback:
run the query, read the database's error, rewrite. It is worth +1 to +10 points
and costs a call only when a query fails.

**The hypothesis this strategy exists to test.** Execution feedback currently
fires only *after* generation is finished. A deep agent moves it *inside*: the
agent can look at the tables, check what a column actually holds, test a join,
and see rows before it commits. If decomposition failed because the stages were
blind, then a decomposition that can see should not fail the same way. If it
loses anyway, decomposition is dead at this model scale and the project will have
shown it a second, independent way.

It can lose. That is the point of building it.

## How it differs from `react`

`react` is the shallow version of this and scored 4.5% generation accuracy at 3B.
Four differences, each aimed at a failure this project has already measured:

**The plan is written down, not held in context.** `react` failed at 3B because
the model could not track a multi-step plan across turns. Here the plan is
external state, rendered back every turn with its progress.

**The prompt does not grow.** `react` appends every tool call and result to a
transcript, so by step six the model is re-reading its own history instead of
working. Here the agent keeps *named notes* and each turn re-renders the current
state — question, plan, what has been learned, what was just tried. Bounded, so
step eight costs what step one costs.

**Decidable work is done in code.** Finding a join path between two tables is a
graph search over the foreign keys. Finding the stored spelling of a value is a
`SELECT DISTINCT`. Neither needs a model, and D8's rule has held every time it
was tested: handing an exact question to a 3B model is how `schema_ok: true`
happens on a hallucinated column.

**Actions are structured output, not the tool-calling API.** `react` discovered
that `qwen2.5-coder:3b` advertises tools and then emits them as plain JSON in the
message content, which cost this project a completely false 95.5%. Structured
decoding constrains the grammar instead, so a malformed action is not
representable. It also frees the loop to re-render state each turn rather than
maintaining a well-formed tool-call transcript.

## Two variants, because the difference is a finding

`deep` starts from nothing. `deep_seeded` starts from `direct`'s draft and
revises it, so it cannot score below the baseline's floor. The pure variant is
the clean architecture comparison; the seeded one is the version anybody would
actually ship. The gap between them says how much of any gain is the agent and
how much is the baseline it was handed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..db.engine import QueryResult, run_query
from ..db.introspect import Schema
from ..llm.client import LLMError
from .base import Draft, Strategy, StrategyContext, strip_sql

MAX_NOTES = 8          # per slot, so the prompt cannot grow without bound
MAX_ROWS_SHOWN = 5
MAX_VALUES = 12


# ── mechanical tools: no model, no chance of hallucination ───────────


def join_path(schema: Schema, start: str, goal: str) -> list[str]:
    """Shortest foreign-key path between two tables, as join conditions.

    A graph search, not a judgement. The orchestrator's join worker is a model
    asked to read foreign keys out of a schema card and state them back; this
    reads the same foreign keys and cannot get them wrong. Phase 3 found wrong
    join keys to be the single most common way a decomposed strategy failed.

    Returns [] when the tables are not connected, which is itself information -
    it means the question needs a table nobody has mentioned yet.
    """
    start, goal = start.lower(), goal.lower()
    edges: dict[str, list[tuple[str, str]]] = {}
    for table in schema.tables:
        for fk in table.foreign_keys:
            left, right = table.name.lower(), fk.ref_table.lower()
            condition = f"{table.name}.{fk.column} = {fk.ref_table}.{fk.ref_column}"
            edges.setdefault(left, []).append((right, condition))
            edges.setdefault(right, []).append((left, condition))

    if start == goal or start not in edges:
        return []

    queue: deque[tuple[str, list[str]]] = deque([(start, [])])
    seen = {start}
    while queue:
        node, path = queue.popleft()
        for neighbour, condition in edges.get(node, []):
            if neighbour in seen:
                continue
            if neighbour == goal:
                return path + [condition]
            seen.add(neighbour)
            queue.append((neighbour, path + [condition]))
    return []


def find_values(
    schema: Schema,
    table: str,
    column: str,
    like: str = "",
    db_url: str | None = None,
) -> list[str]:
    """The values a column actually holds, closest match first.

    This is the failure the schema card was built for and still does not fully
    solve: a model filtering `status = 'Shipped'` when the column holds
    `'shipped'` gets valid SQL, zero rows, and no error to learn from. The
    schema card samples three values per categorical column; this lets the agent
    ask about any column, at the moment it is about to filter on one.
    """
    if not _exists(schema, table, column):
        return []

    result = run_query(
        f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL',
        db_url=db_url,
        max_rows=200,
    )
    if not result.ok:
        return []

    values = [str(row[0]) for row in result.rows]
    if like:
        import difflib

        ranked = difflib.get_close_matches(like, values, n=MAX_VALUES, cutoff=0.4)
        remaining = [v for v in values if v not in ranked]
        values = ranked + remaining
    return values[:MAX_VALUES]


def diagnose_empty_result(
    sql: str,
    schema: Schema,
    dialect: str = "sqlite",
    db_url: str | None = None,
) -> list[str]:
    """Why a valid query matched nothing, answered without a model.

    Zero rows is the failure with no error attached: the SQL is legal, the
    database is content, and the answer is wrong. The usual cause is a string
    literal that does not match the stored spelling - `'Shipped'` against a
    column holding `'shipped'`.

    Measured on the 3B model: told "0 rows, check the values" with `find_values`
    available, the agent re-ran the identical query twice rather than calling it.
    Telling an agent what to check is a request; checking it is a control. This
    reads the literals straight out of the parsed query, looks each one up, and
    reports the spelling that is actually there.
    """
    import sqlglot
    from sqlglot import exp

    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except Exception:
        return []
    if tree is None:
        return []

    # Resolve aliases, so `o.status` finds its way back to `orders`.
    tables = {t.name.lower(): t.name for t in schema.tables}
    aliases: dict[str, str] = {}
    for node in tree.find_all(exp.Table):
        real = tables.get((node.name or "").lower())
        if not real:
            continue
        aliases[real.lower()] = real
        if node.alias:
            aliases[node.alias.lower()] = real

    findings: list[str] = []
    for eq in tree.find_all(exp.EQ):
        column = next((side for side in (eq.left, eq.right) if isinstance(side, exp.Column)), None)
        literal = next(
            (side for side in (eq.left, eq.right)
             if isinstance(side, exp.Literal) and side.is_string),
            None,
        )
        if column is None or literal is None:
            continue

        written = literal.this
        owner = aliases.get((column.table or "").lower())
        candidates = [owner] if owner else list(tables.values())
        for table in candidates:
            values = find_values(schema, table, column.name, like=written, db_url=db_url)
            if not values:
                continue
            if written in values:
                break  # spelled correctly; the empty result is something else
            finding = (
                f"{table}.{column.name} = '{written}' matches nothing. "
                f"Stored values include: {', '.join(repr(v) for v in values[:5])}"
            )
            if finding not in findings:
                findings.append(finding)
            break

    return findings


def _exists(schema: Schema, table: str, column: str) -> bool:
    for t in schema.tables:
        if t.name.lower() == table.lower():
            return any(c.name.lower() == column.lower() for c in t.columns)
    return False


# ── the agent's state, kept in named slots rather than a transcript ──


@dataclass
class Scratchpad:
    """What the agent has learned, bounded.

    Named slots rather than an append-only conversation. This is the whole
    reason the prompt does not grow with the number of steps: turn eight
    re-renders the same four lists that turn one did.
    """

    plan: list[str] = field(default_factory=list)
    done: int = 0
    schema_notes: list[str] = field(default_factory=list)
    join_notes: list[str] = field(default_factory=list)
    value_notes: list[str] = field(default_factory=list)
    attempts: list[tuple[str, str]] = field(default_factory=list)  # (sql, outcome)

    def note(self, slot: list[str], text: str) -> None:
        if text and text not in slot:
            slot.append(text)
            del slot[:-MAX_NOTES]

    def render(self) -> str:
        parts = []
        if self.plan:
            steps = [
                f"  [{'x' if i < self.done else ' '}] {i + 1}. {step}"
                for i, step in enumerate(self.plan)
            ]
            parts.append("Plan:\n" + "\n".join(steps))

        for title, notes in (
            ("Tables and columns confirmed", self.schema_notes),
            ("Join conditions (read from the foreign keys, these are exact)", self.join_notes),
            ("Values as actually stored", self.value_notes),
        ):
            if notes:
                parts.append(f"{title}:\n" + "\n".join(f"  - {n}" for n in notes))

        if self.attempts:
            rendered = "\n".join(
                f"  {sql}\n    -> {outcome}" for sql, outcome in self.attempts[-3:]
            )
            parts.append(f"Queries already tried:\n{rendered}")

        return "\n\n".join(parts) if parts else "(nothing learned yet)"


# ── the action schema ────────────────────────────────────────────────


class Action(BaseModel):
    """One step. Flat on purpose.

    Nested or union schemas are where constrained decoding gets unreliable on
    small models, so every argument any tool might need is a top-level optional
    field and the unused ones stay empty.
    """

    thought: str = Field(default="", description="One sentence on why this step.")
    tool: Literal[
        "inspect_tables", "find_join_path", "find_values", "try_sql", "submit"
    ] = Field(description="The action to take.")
    tables: list[str] = Field(default_factory=list, description="For inspect_tables.")
    from_table: str = Field(default="", description="For find_join_path.")
    to_table: str = Field(default="", description="For find_join_path.")
    table: str = Field(default="", description="For find_values.")
    column: str = Field(default="", description="For find_values.")
    like: str = Field(default="", description="For find_values: the wording in the question.")
    sql: str = Field(default="", description="For try_sql and submit.")
    completed_step: int = Field(default=0, description="Plan step just finished, or 0.")


class Plan(BaseModel):
    """Three to five steps, written once and then tracked."""

    steps: list[str] = Field(default_factory=list, description="Short imperative steps.")


PLAN_SYSTEM = """You plan how to answer a question with SQL against a database \
you cannot see yet.

Write three to five short steps. Each step is something concrete to find out or \
to do - which tables hold the data, how they join, what a filter value is \
actually spelled as, then write and test the query.

Do not write SQL here. Do not invent table names."""

SYSTEM = """You answer questions with {dialect} SQL by investigating the database \
before you commit.

You cannot see the database directly. Each turn you choose ONE action:

  inspect_tables   - see the columns of specific tables. Do this before using them.
  find_join_path   - get the exact join condition between two tables, read from
                     the foreign keys. Always prefer this to guessing a join.
  find_values      - see what a column actually contains. Do this before
                     filtering on a text value: the question may say "shipped"
                     while the column holds "Shipped", which returns zero rows
                     and no error.
  try_sql          - run a query and see what comes back. Cheap. Use it.
  submit           - finish, with your final query.

Rules:
- Never reference a table or column you have not confirmed.
- Test before you submit. A query that runs and returns plausible rows is worth
  far more than one you believe in.
- If a query errors, read the error and fix that specific thing.
- Do not repeat an action you have already taken; the notes show what you know.
- Write exactly one SELECT statement."""

USER = """Question: {question}

{state}

Tables in this database:
{tables}

{budget}
Choose your next action."""


class DeepAgentStrategy(Strategy):
    """An agent that investigates the database before committing to a query."""

    name = "deep"
    description = "Plans, inspects the database, tests queries, then submits."
    seeded = False

    def __init__(self, max_steps: int = 8, plan_first: bool = True):
        self.max_steps = max_steps
        self.plan_first = plan_first

    # ── entry point ──────────────────────────────────────────────────

    def generate(self, ctx: StrategyContext) -> Draft:
        pad = Scratchpad()
        client = ctx.client("sql")
        best_sql, submitted = "", ""

        with ctx.trace.span("deep-agent", "investigating") as agent_span:
            if self.seeded:
                best_sql = self._seed(ctx, pad)

            if self.plan_first:
                self._make_plan(ctx, pad)

            steps = 0
            seen: dict[str, str] = {}
            stalled = 0

            for turn in range(self.max_steps):
                left = self.max_steps - turn
                budget = f"You have {left} step{'s' if left != 1 else ''} left."
                if left <= 2:
                    budget += (
                        " Submit your best query now, even if you are not certain."
                        " An unsubmitted query scores nothing."
                    )
                if stalled >= 2:
                    budget += (
                        " You have repeated yourself. Stop gathering information"
                        " and either try_sql or submit."
                    )

                try:
                    action = client.structured(
                        SYSTEM.format(dialect=ctx.dialect),
                        USER.format(
                            question=ctx.question,
                            state=pad.render(),
                            tables=ctx.schema.render_compact(),
                            budget=budget,
                        ),
                        Action,
                    )
                except LLMError as e:
                    agent_span.fail(f"agent loop failed: {e}")
                    break

                steps += 1
                if action.completed_step > pad.done:
                    pad.done = min(action.completed_step, len(pad.plan))

                if action.tool == "submit":
                    submitted = strip_sql(action.sql)
                    with ctx.trace.span("submit", "final answer") as span:
                        span.finish(sql=submitted)
                    break

                # An action identical to one already taken cannot teach the
                # agent anything, and repeating it is how `react` lost whole
                # questions at 3B. Answer it with what it produced last time and
                # push for something else.
                signature = _signature(action)
                if signature in seen:
                    stalled += 1
                    outcome, ran_ok = (
                        f"You already did exactly this. It returned: {seen[signature]}"
                        " Do something different.",
                        False,
                    )
                else:
                    stalled = 0
                    outcome, ran_ok = self._dispatch(ctx, pad, action)
                    seen[signature] = outcome[:160]

                if action.tool == "try_sql" and ran_ok:
                    best_sql = strip_sql(action.sql)

                with ctx.trace.span(action.tool, action.thought[:60]) as span:
                    span.finish(result=outcome[:200])

            agent_span.finish(steps=steps, submitted=bool(submitted))

        # A submitted query wins. Otherwise fall back to the best query that
        # actually ran, and only then to nothing - an agent that explored for
        # eight steps and produced no SQL should not be graded as if the repair
        # layer's rescue were its own work. That was `react`'s false 95.5%.
        sql = submitted or best_sql
        return Draft(
            sql=sql,
            notes={
                "steps": steps,
                "submitted": bool(submitted),
                "tested": bool(best_sql),
                "seeded": self.seeded,
            },
        )

    # ── setup ────────────────────────────────────────────────────────

    def _seed(self, ctx: StrategyContext, pad: Scratchpad) -> str:
        """Start from `direct`'s draft, so the floor is the baseline's answer."""
        from .direct import DirectStrategy

        draft = DirectStrategy().generate(ctx)
        if not draft.sql:
            return ""
        result = run_query(draft.sql, db_url=ctx.db_url)
        pad.attempts.append((_one_line(draft.sql), _describe(result)))
        return draft.sql if result.ok else ""

    def _make_plan(self, ctx: StrategyContext, pad: Scratchpad) -> None:
        with ctx.trace.span("planner", "writing a plan") as span:
            try:
                plan = ctx.client("critic").structured(
                    PLAN_SYSTEM,
                    f"Question: {ctx.question}\n\n"
                    f"Tables available:\n{ctx.schema.render_compact()}",
                    Plan,
                )
                pad.plan = [s for s in plan.steps if s.strip()][:5]
            except LLMError as e:
                # A missing plan degrades the agent to a plain tool loop rather
                # than ending the question.
                span.fail(str(e))
            span.finish(steps=len(pad.plan))

    # ── tools ────────────────────────────────────────────────────────

    def _dispatch(
        self, ctx: StrategyContext, pad: Scratchpad, action: Action
    ) -> tuple[str, bool]:
        """Run one action. Returns (what to record, whether SQL executed)."""
        if action.tool == "inspect_tables":
            if not action.tables:
                return (
                    "inspect_tables needs `tables`. Name one or more of: "
                    f"{', '.join(ctx.schema.table_names)}",
                    False,
                )
            subset = ctx.schema.subset(action.tables)
            if not subset.tables:
                return (
                    f"No such table. Available: {', '.join(ctx.schema.table_names)}",
                    False,
                )
            for table in subset.tables:
                pad.note(pad.schema_notes, table.render().replace("\n", "\n    "))
            return f"inspected {', '.join(t.name for t in subset.tables)}", False

        if action.tool == "find_join_path":
            # A model that picks the tool but leaves the arguments empty must be
            # told that, not handed a plausible-sounding "no path between '' and
            # ''" - which is what let a 3B model spend seven consecutive steps
            # asking the same empty question.
            missing = [
                f for f, v in (("from_table", action.from_table),
                               ("to_table", action.to_table)) if not v.strip()
            ]
            if missing:
                return (
                    f"find_join_path needs {' and '.join(missing)}. "
                    f"Both must be table names from: {', '.join(ctx.schema.table_names)}",
                    False,
                )
            conditions = join_path(ctx.schema, action.from_table, action.to_table)
            if not conditions:
                note = (
                    f"No foreign-key path between '{action.from_table}' and "
                    f"'{action.to_table}'. They may need a table in between."
                )
                pad.note(pad.join_notes, note)
                return note, False
            for condition in conditions:
                pad.note(pad.join_notes, condition)
            return " AND ".join(conditions), False

        if action.tool == "find_values":
            if not action.table.strip() or not action.column.strip():
                return "find_values needs both `table` and `column`.", False
            values = find_values(
                ctx.schema, action.table, action.column, action.like, ctx.db_url
            )
            if not values:
                note = f"{action.table}.{action.column} is not a column with readable values."
                return note, False
            rendered = ", ".join(repr(v) for v in values)
            pad.note(pad.value_notes, f"{action.table}.{action.column}: {rendered}")
            return rendered, False

        if action.tool == "try_sql":
            sql = strip_sql(action.sql)
            if not sql:
                return "No query given.", False
            result = run_query(sql, db_url=ctx.db_url)
            outcome = _describe(result)

            if result.ok and not result.rows:
                # Do not ask the agent to go and check the values. Check them.
                for finding in diagnose_empty_result(
                    sql, ctx.schema, ctx.dialect, ctx.db_url
                ):
                    pad.note(pad.value_notes, finding)
                    outcome += "\n    " + finding

            pad.attempts.append((_one_line(sql), outcome))
            del pad.attempts[:-MAX_NOTES]
            return outcome, result.ok

        return f"Unknown action '{action.tool}'.", False


class DeepSeededStrategy(DeepAgentStrategy):
    """The deep agent, handed `direct`'s draft to start from.

    Cannot score below the baseline's floor, which makes it the version worth
    shipping - and makes the gap to `deep` the measure of how much of any gain
    is the agent rather than the baseline it was given.
    """

    name = "deep_seeded"
    description = "The deep agent, starting from the single-call baseline's draft."
    seeded = True


# ── helpers ──────────────────────────────────────────────────────────


def _describe(result: QueryResult) -> str:
    """What a query attempt should tell the agent."""
    if not result.ok:
        return f"ERROR: {result.error}"
    if not result.rows:
        return "0 rows. Valid SQL, but the filters matched nothing - check the values."
    shown = result.rows[:MAX_ROWS_SHOWN]
    body = "; ".join(str(tuple(r)) for r in shown)
    more = f" (+{len(result.rows) - len(shown)} more)" if len(result.rows) > len(shown) else ""
    return f"{result.row_count} rows: {body}{more}"


def _signature(action: Action) -> str:
    """What makes two actions the same action."""
    return "|".join([
        action.tool,
        ",".join(sorted(t.lower() for t in action.tables)),
        action.from_table.lower(), action.to_table.lower(),
        action.table.lower(), action.column.lower(), action.like.lower(),
        _one_line(action.sql).lower(),
    ])


def _one_line(sql: str) -> str:
    return " ".join(sql.split())
