"""Command line entry point.

    python -m aqueduct.cli ask "how many employees are in each department?"
    python -m aqueduct.cli seed
    python -m aqueduct.cli schema
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table as RichTable

from .config import settings
from .observability.trace import Span, Status

app = typer.Typer(add_completion=False, help="Aqueduct - an agentic Text-to-SQL crew.")
console = Console()

_STATUS_STYLE = {
    Status.DONE: ("green", "done"),
    Status.FAILED: ("red", "failed"),
    Status.RUNNING: ("yellow", "running"),
    Status.SKIPPED: ("dim", "skipped"),
}


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to answer."),
    explain: bool = typer.Option(False, "--explain", "-e", help="Add a plain-English answer."),
    show_trace: bool = typer.Option(True, "--trace/--no-trace", help="Show which agents ran."),
) -> None:
    """Ask the crew a question."""
    from .crew import Crew

    console.print(Panel(question, title="question", border_style="cyan"))

    with console.status("[dim]crew working...", spinner="dots"):
        answer = Crew().ask(question, explain=explain)

    console.print(Syntax(answer.sql, "sql", theme="ansi_dark", word_wrap=True))

    if answer.ok:
        console.print(_result_table(answer.result))
    else:
        console.print(Panel(str(answer.result.error), title="failed", border_style="red"))

    if answer.explanation:
        console.print(Panel(answer.explanation, title="answer", border_style="green"))

    if show_trace:
        console.print()
        _print_trace(answer.trace.root)

    console.print(f"\n[dim]{answer.usage.summary()}[/dim]")
    raise typer.Exit(0 if answer.ok else 1)


@app.command()
def seed() -> None:
    """Create and load the demo database."""
    from .db.seed import seed as run_seed

    counts = run_seed()
    console.print(f"[green]Seeded[/green] {settings.db_url}")
    for table, n in counts.items():
        console.print(f"  {table:<14} {n:>4} rows")


@app.command()
def schema(compact: bool = typer.Option(False, "--compact", "-c")) -> None:
    """Print the schema exactly as the agents see it."""
    from .db.introspect import load_schema

    loaded = load_schema()
    console.print(loaded.render_compact() if compact else loaded.render())


@app.command()
def doctor() -> None:
    """Check that the database and the LLM backend are both reachable."""
    from .db.engine import run_query
    from .llm.client import LLMClient

    ok = True

    probe = run_query("SELECT 1 AS ok")
    if probe.ok:
        console.print(f"[green]OK[/green]   database  {settings.db_url}")
    else:
        ok = False
        console.print(f"[red]FAIL[/red] database  {probe.error}")

    try:
        reply = LLMClient(role="sql").chat("Reply with the word ready.", "ready?")
        console.print(
            f"[green]OK[/green]   llm       {settings.model_sql} "
            f"at {settings.base_url} -> {reply[:40]!r}"
        )
    except Exception as e:
        ok = False
        console.print(f"[red]FAIL[/red] llm       {e}")

    raise typer.Exit(0 if ok else 1)


@app.command()
def cache(clear: bool = typer.Option(False, "--clear", help="Delete all cached responses.")) -> None:
    """Inspect or clear the LLM response cache."""
    from .llm import cache as llm_cache

    if clear:
        console.print(f"Removed {llm_cache.clear()} cached responses.")
        return
    s = llm_cache.stats()
    console.print(f"{s['entries']} entries, {s['bytes'] / 1024:.1f} KiB")


def _result_table(result) -> RichTable:
    table = RichTable(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    for column in result.columns:
        table.add_column(str(column))
    for row in result.rows[:25]:
        table.add_row(*("NULL" if v is None else str(v) for v in row))
    if result.row_count > 25:
        table.caption = f"{result.row_count - 25} more rows"
    elif result.row_count == 0:
        table.caption = "no rows matched"
    return table


def _print_trace(span: Span, depth: int = 0) -> None:
    """Render the agent tree — who was spun up, and how long each took."""
    if span.agent != "crew":
        style, label = _STATUS_STYLE[span.status]
        indent = "  " * (depth - 1)
        console.print(
            f"{indent}[{style}]*[/{style}] [bold]{span.agent}[/bold] "
            f"{span.label} [dim]{span.elapsed_ms:.0f}ms {label}[/dim]"
        )
        if span.status is Status.FAILED and "error" in span.detail:
            console.print(f"{indent}    [red]{span.detail['error']}[/red]")
    for child in span.children:
        _print_trace(child, depth + 1)


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
