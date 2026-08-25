"""ReAct — the tool-calling agent from notebook 1.

The only strategy here that is not handed the schema. It gets three tools —
`list_tables`, `get_table_schema`, `run_sql` — and has to discover the database
for itself, deciding at each step what to look at next.

That makes it the honest comparison for the others. Every other strategy in this
project starts with a fully introspected schema card in its prompt; this one has
to earn that information, and the difference in cost and accuracy is exactly what
tells you whether schema introspection up front is worth doing.

**It is also the strategy most likely to fail on a small model.** Multi-step tool
use requires the model to track what it has already learned and choose a next
action; 3B models lose that thread. The loop is therefore written defensively —
step budget, repeated-call detection, and a fallback that salvages the last query
attempted rather than returning nothing.
"""

from __future__ import annotations

import json

from ..db.engine import run_query
from ..llm.client import LLMError
from .base import Draft, Strategy, StrategyContext, strip_sql

SYSTEM_PROMPT = """You are a SQL analyst with access to a {dialect} database.

You cannot see the database. Use the tools to explore it:
  1. list_tables       - see what tables exist
  2. get_table_schema  - see the columns of a table before using it
  3. run_sql           - run your SELECT query

Work step by step. Never reference a table or column you have not confirmed with \
a tool. When run_sql returns the answer, reply with a short summary - do not call \
more tools.

Only SELECT queries. Never write to the database."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List every table in the database. Call this first.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_schema",
            "description": "Get the columns of one table. Call before querying it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Exact table name"}
                },
                "required": ["table_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a SELECT query and return the rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A single SELECT statement"}
                },
                "required": ["query"],
            },
        },
    },
]


class ReactStrategy(Strategy):
    name = "react"
    description = "Tool-calling agent that discovers the schema for itself."

    def __init__(self, max_steps: int = 8):
        self.max_steps = max_steps

    def generate(self, ctx: StrategyContext) -> Draft:
        client = ctx.client("sql")
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(dialect=ctx.dialect)},
            {"role": "user", "content": ctx.question},
        ]

        last_sql = ""
        successful_sql = ""
        seen_calls: set[str] = set()
        steps = 0

        with ctx.trace.span("react-agent", "exploring the database") as agent_span:
            for _ in range(self.max_steps):
                try:
                    response = client._client.chat.completions.create(
                        model=client.model,
                        temperature=client.temperature,
                        messages=messages,
                        tools=TOOLS,
                        tool_choice="auto",
                    )
                except Exception as e:
                    agent_span.fail(f"tool loop failed: {e}")
                    break

                client.usage.record(client.model, response.usage, 0.0, was_cached=False)
                message = response.choices[0].message
                messages.append(message.model_dump(exclude_none=True))

                if not message.tool_calls:
                    break  # the agent considers itself done

                for call in message.tool_calls:
                    steps += 1
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    # A model looping on the same call is stuck, not thinking.
                    signature = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if signature in seen_calls and name != "run_sql":
                        output = "Already called with these arguments. Use what you have."
                    else:
                        seen_calls.add(signature)
                        output = self._dispatch(ctx, name, args)

                    if name == "run_sql":
                        last_sql = args.get("query", "") or last_sql
                        if not output.startswith("ERROR"):
                            successful_sql = last_sql

                    with ctx.trace.span(name, _describe(name, args)) as tool_span:
                        tool_span.finish(result=output[:200])

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": output[:4000],
                        }
                    )

                if successful_sql:
                    break

            agent_span.finish(steps=steps, sql=successful_sql or last_sql)

        # Prefer a query that ran. Fall back to the last attempt so a failed
        # exploration still produces something the Crew's repair loop can work
        # with, rather than an empty string that grades as a null answer.
        sql = successful_sql or last_sql
        return Draft(sql=strip_sql(sql), notes={"steps": steps, "executed": bool(successful_sql)})

    def _dispatch(self, ctx: StrategyContext, name: str, args: dict) -> str:
        """Run a tool. Errors are returned as text — they are what the agent learns from."""
        if name == "list_tables":
            return ", ".join(ctx.schema.table_names)

        if name == "get_table_schema":
            table = str(args.get("table_name", ""))
            subset = ctx.schema.subset([table])
            if not subset.tables:
                return (
                    f"ERROR: no table named '{table}'. "
                    f"Available: {', '.join(ctx.schema.table_names)}"
                )
            return subset.render()

        if name == "run_sql":
            query = str(args.get("query", ""))
            if not query.strip():
                return "ERROR: no query provided."
            result = run_query(query)
            if not result.ok:
                return f"ERROR: {result.error}"
            return result.to_markdown(max_rows=10)

        return f"ERROR: unknown tool '{name}'."


def _describe(name: str, args: dict) -> str:
    if name == "get_table_schema":
        return f"inspecting {args.get('table_name', '?')}"
    if name == "run_sql":
        return "running query"
    return "listing tables"
