"""Execution trace.

Every agent that runs opens a span and closes it. The result is a tree showing
who was spun up, in what order, how long each took, and what they produced.

This exists for three separate reasons, which is why it is in from the start
rather than bolted on:

  * the UI renders agent cards directly from it — the live view *is* the trace;
  * the evaluation harness counts calls and latency per agent from it;
  * when a wrong answer comes back, the trace is how you find out which agent
    made the mistake instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any, Iterator


class Status(str, Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Span:
    """One agent's turn."""

    agent: str
    label: str = ""
    status: Status = Status.RUNNING
    detail: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    started: float = field(default_factory=perf_counter)
    ended: float | None = None

    @property
    def elapsed_ms(self) -> float:
        return ((self.ended or perf_counter()) - self.started) * 1000

    def finish(self, status: Status = Status.DONE, **detail: Any) -> "Span":
        self.status = status
        self.detail.update(detail)
        self.ended = perf_counter()
        return self

    def fail(self, reason: str, **detail: Any) -> "Span":
        return self.finish(Status.FAILED, error=reason, **detail)

    def walk(self) -> Iterator["Span"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "label": self.label,
            "status": self.status.value,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "detail": self.detail,
            "children": [c.to_dict() for c in self.children],
        }


class Trace:
    """Collects spans for one question.

    Spans nest via `with trace.span(...)`, so an agent that spins up sub-agents
    produces a tree rather than a flat list — which is what makes "the Lead spun
    up three critics" visible instead of inferred.
    """

    def __init__(self, question: str = ""):
        self.question = question
        self.root = Span(agent="crew", label=question, status=Status.RUNNING)
        self._stack: list[Span] = [self.root]
        self._listeners: list = []

    def on_update(self, callback) -> None:
        """Register a callback fired whenever a span opens or closes.

        Streamlit uses this to redraw agent cards live rather than waiting for
        the whole crew to finish.
        """
        self._listeners.append(callback)

    def _notify(self) -> None:
        for callback in self._listeners:
            try:
                callback(self)
            except Exception:
                pass  # a broken listener must never take down the crew

    def span(self, agent: str, label: str = "", **detail: Any) -> "_SpanContext":
        return _SpanContext(self, agent, label, detail)

    def finish(self, status: Status = Status.DONE, **detail: Any) -> None:
        self.root.finish(status, **detail)
        self._notify()

    @property
    def agents_used(self) -> list[str]:
        """Distinct agents that ran, in first-appearance order.

        This is the headcount number — the thing the router is trying to keep
        small on easy questions.
        """
        seen: list[str] = []
        for span in self.root.walk():
            if span.agent != "crew" and span.agent not in seen:
                seen.append(span.agent)
        return seen

    @property
    def span_count(self) -> int:
        return sum(1 for s in self.root.walk() if s.agent != "crew")

    def to_dict(self) -> dict:
        return self.root.to_dict()


class _SpanContext:
    """Context manager returned by `Trace.span`."""

    def __init__(self, trace: Trace, agent: str, label: str, detail: dict):
        self.trace = trace
        self.span = Span(agent=agent, label=label, detail=dict(detail))

    def __enter__(self) -> Span:
        self.trace._stack[-1].children.append(self.span)
        self.trace._stack.append(self.span)
        self.trace._notify()
        return self.span

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.span.fail(f"{exc_type.__name__}: {exc}")
        elif self.span.status is Status.RUNNING:
            self.span.finish()
        self.trace._stack.pop()
        self.trace._notify()
        return False  # never swallow exceptions
