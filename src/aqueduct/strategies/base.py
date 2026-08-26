"""The Strategy interface.

A strategy is *one way to turn a question into SQL*. That is the whole contract —
it does not execute, grade, or repair. Those live in the Crew, deliberately:

  * Phase 2 established that execution feedback is worth +4.5 points. If each
    strategy carried its own repair loop, every comparison between strategies
    would be contaminated by differences in how they repair.
  * Holding repair constant means a difference between two strategies is
    attributable to *generation*, which is what we are actually trying to measure.

Where a strategy has verification built into its definition — prompt chaining's
verify stage, the evaluator-optimizer's loop — that stays inside the strategy,
because removing it would no longer be that strategy. The distinction is between
verification that *defines* the pattern and repair that is bolted on afterwards.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..db.introspect import Schema
from ..llm.client import LLMClient, Usage
from ..observability.trace import Trace


@dataclass
class StrategyContext:
    """Everything a strategy is given to work with."""

    question: str
    schema: Schema
    trace: Trace
    usage: Usage
    dialect: str = "sqlite"
    memory_context: str = ""
    # BIRD gives every question its own database, so the target cannot be a
    # global setting.
    db_url: str | None = None

    def client(self, role: str) -> LLMClient:
        """A client for one agent role, sharing this run's usage counter."""
        return LLMClient(role=role, usage=self.usage)


@dataclass
class Draft:
    """A strategy's output."""

    sql: str
    notes: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.sql


class Strategy(ABC):
    """Base class for every generation strategy."""

    name: str = "strategy"
    description: str = ""

    @abstractmethod
    def generate(self, ctx: StrategyContext) -> Draft:
        """Produce SQL for `ctx.question`."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


def strip_sql(text: str) -> str:
    """Clean a model's SQL response.

    Structured decoding covers JSON, but SQL comes back as free text, so the
    habit of wrapping it in fences survives and has to be undone.
    """
    from ..db.safety import _strip_fences

    cleaned = _strip_fences(text).strip()

    # Models sometimes prefix a label even when told not to.
    for prefix in ("sql:", "SQL:", "Query:", "query:"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()

    return cleaned
