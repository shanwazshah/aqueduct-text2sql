"""Aqueduct — the demo UI.

    streamlit run ui/app.py

The point of this page is to make the project's finding *visible*. Reading that
`orchestrator` spends six LLM calls to score nine points below a single call is
one thing; watching seven agent cards appear one after another, wait, and then
lose to a strategy that finished in one step is another.

So the layout is deliberate: the question and answer on the left, and on the
right the crew as it actually runs — each agent appearing when it starts, with
its own timing, and the running cost underneath.

**On live updates.** Streamlit re-runs the whole script on every interaction, so
there is no natural way to stream progress from a long call. The crew therefore
runs on a worker thread while the main thread polls the trace it is filling in
and repaints. That is why `Crew.ask` accepts a trace from outside — otherwise the
span tree only becomes reachable once the run is over, which is too late to show
anything happening.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aqueduct.config import settings                      # noqa: E402
from aqueduct.crew import Answer, Crew, RepairMode        # noqa: E402
from aqueduct.db.introspect import load_schema            # noqa: E402
from aqueduct.observability.trace import Span, Status, Trace  # noqa: E402
from aqueduct.strategies import STRATEGIES                # noqa: E402

st.set_page_config(page_title="Aqueduct", page_icon="🚰", layout="wide")

# Shown in the picker so the cost of each strategy is visible at the moment it is
# chosen, not afterwards.
#
# These are two different measurements and they are kept apart deliberately.
# Presenting a demo-set number under a BIRD label is the mistake this project
# already retracted once, and a sidebar is exactly where it would slip back in.
#
#   bird_ex     gen EX on BIRD mini-dev, 100 questions, 7B. Only `direct`,
#               `chain` and `orchestrator` were ever run there; the rest show a
#               dash rather than a borrowed score.
#   demo_calls  mean LLM calls per question on the 22-question demo set at 3B,
#               which is where every strategy has been run. A cost figure, not a
#               score - and demo-set cost, not BIRD cost.
BENCHMARK = {
    "direct":        {"bird_ex": 41.0, "demo_calls": 1.0, "note": "the baseline that wins"},
    "chain":         {"bird_ex": 35.0, "demo_calls": 4.5, "note": "5-stage pipeline"},
    "orchestrator":  {"bird_ex": 32.0, "demo_calls": 5.8, "note": "planner + specialists"},
    "parallel":      {"bird_ex": None, "demo_calls": 3.7, "note": "critics vote"},
    "eval_optimize": {"bird_ex": None, "demo_calls": 5.3, "note": "grade and revise"},
    "react":         {"bird_ex": None, "demo_calls": 2.5, "note": "tool-using agent"},
}

STATUS_ICON = {
    Status.RUNNING: "⏳",
    Status.DONE: "✅",
    Status.FAILED: "❌",
    Status.SKIPPED: "⏭️",
}


@st.cache_resource(show_spinner=False)
def get_schema():
    """Introspect once per process — it costs a query per text column."""
    return load_schema()


def _looks_like_backend_down(error: Exception) -> bool:
    """Is this 'the model server is not running' rather than a real failure?"""
    text = str(error).lower()
    return any(
        phrase in text
        for phrase in ("connection error", "connection refused", "failed to connect",
                       "max retries", "cannot connect", "actively refused")
    )


def backend_is_up() -> bool:
    """Cheap liveness probe, so the page can warn before a question is asked."""
    import urllib.error
    import urllib.request

    root = settings.base_url.rsplit("/v1", 1)[0]
    try:
        urllib.request.urlopen(f"{root}/api/version", timeout=2)
        return True
    except Exception:
        return False


def render_agents(container, trace: Trace) -> None:
    """Draw the crew as it stands right now."""
    spans = [s for s in trace.root.walk() if s.agent != "crew"]

    with container.container():
        if not spans:
            st.caption("waiting for the first agent…")
            return

        for span in spans:
            icon = STATUS_ICON.get(span.status, "•")
            # Sub-second spans are either a cache hit or the database, and
            # both are real. Printing "0.0s" makes a working run look broken, so
            # show milliseconds instead of rounding the truth away.
            elapsed = (
                f"{span.elapsed_ms:.0f}ms" if span.elapsed_ms < 1000
                else f"{span.elapsed_ms / 1000:.1f}s"
            )
            st.markdown(f"**{icon} {span.agent}** · {span.label}  \n`{elapsed}`")

            if span.status is Status.FAILED and "error" in span.detail:
                st.error(span.detail["error"], icon="⚠️")

            sql = span.detail.get("sql")
            if sql:
                st.code(sql, language="sql")

            for key in ("tier", "score", "signals", "issues", "errors", "workers",
                        "tables", "objections", "rows", "verdict"):
                value = span.detail.get(key)
                if value:
                    st.caption(f"{key}: {value}")


def run_live(crew: Crew, question: str, agents_slot, cost_slot,
             explain: bool = False) -> Answer:
    """Run the crew on a worker thread, repainting the trace while it works."""
    trace = Trace(question)
    box: dict = {}

    def work():
        try:
            box["answer"] = crew.ask(question, trace=trace, explain=explain)
        except Exception as e:  # surfaced in the UI rather than swallowed
            box["error"] = e

    def paint_cost():
        usage = crew.usage
        parts = [f"{usage.calls} call{'s' if usage.calls != 1 else ''}"]
        if usage.cached:
            parts.append(f"{usage.cached} cached")
        if usage.total_tokens:
            parts.append(f"{usage.total_tokens} tokens")
        parts.append(f"{usage.seconds:.1f}s")
        cost_slot.caption(" · ".join(parts))

    thread = threading.Thread(target=work, daemon=True)
    thread.start()

    while thread.is_alive():
        render_agents(agents_slot, trace)
        paint_cost()
        time.sleep(0.4)

    thread.join()
    # Repaint after the join as well. A fully cached run finishes before the
    # first poll, so without this the panel keeps the zeros it started with.
    render_agents(agents_slot, trace)
    paint_cost()

    if "error" in box:
        raise box["error"]
    return box["answer"]


# ── sidebar ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚰 Aqueduct")
    st.caption("Agentic Text-to-SQL, with the receipts.")

    strategy = st.selectbox(
        "Strategy",
        list(STRATEGIES),
        format_func=lambda n: f"{n} — {BENCHMARK[n]['note']}",
    )

    info = BENCHMARK[strategy]
    left, right = st.columns(2)
    left.metric("BIRD gen EX", f"{info['bird_ex']:.0f}%" if info["bird_ex"] else "—")
    right.metric("calls/q (demo set)", f"{info['demo_calls']:.1f}")
    if info["bird_ex"] is None:
        st.caption("Not run on BIRD - no score to show.")

    repair = st.select_slider(
        "Repair",
        options=[RepairMode.NONE, RepairMode.EXECUTION, RepairMode.BOTH],
        value=RepairMode.EXECUTION,
        format_func=lambda m: {"none": "off", "execution": "on error",
                               "both": "error + critic"}[m.value],
    )
    st.caption(
        "Execution repair was worth **+1 to +10 points** and costs a call only "
        "when a query fails. Adding the critic bought **+0.0** for double the "
        "calls."
    )

    explain = st.toggle("Explain the answer in English", value=True)

    st.divider()
    st.caption(f"**model** `{settings.model_sql}`")
    # Splitting on "/" alone leaves the whole Windows path on screen.
    st.caption(f"**database** `{Path(settings.db_url.split('///')[-1]).name}`")

    with st.expander("Schema"):
        st.code(get_schema().render_compact(), language="text")


# ── main ─────────────────────────────────────────────────────────────

st.markdown("#### Ask the database a question")

# Warn before a question is typed rather than after 13 seconds of waiting.
if not backend_is_up():
    st.warning(
        f"The model server at `{settings.base_url}` is not responding. "
        "Start it with `ollama serve`, then reload — questions will fail until "
        "you do.",
        icon="🔌",
    )

question = st.text_input(
    "question",
    placeholder="Which product category generated the most revenue?",
    label_visibility="collapsed",
)

examples = [
    "How many employees are in each department?",
    "Which product category generated the most revenue?",
    "Which employees have never handled an order?",
    "For each department, who is the highest paid employee?",
]
cols = st.columns(len(examples))
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        question = example

if question:
    answer_col, crew_col = st.columns([3, 2], gap="large")

    with crew_col:
        st.markdown("##### The crew")
        agents_slot = st.empty()
        cost_slot = st.empty()

    with answer_col:
        crew = Crew(strategy=strategy, repair=repair, schema=get_schema(),
                    use_memory=False)
        try:
            answer = run_live(crew, question, agents_slot, cost_slot, explain=explain)
        except Exception as e:
            # "Connection error" is what the OpenAI client says when nothing is
            # listening, which tells the reader nothing actionable. Ollama stops
            # on its own — an auto-update restarts the service — so this is the
            # most likely failure on this page and deserves the real answer.
            if _looks_like_backend_down(e):
                st.error(f"Can't reach the model server at `{settings.base_url}`.")
                st.markdown(
                    "**Ollama is not running.** Start it in a terminal:\n\n"
                    "```\nollama serve\n```\n\n"
                    "Then reload this page. Ollama stops itself after an "
                    "auto-update, so this happens occasionally."
                )
            else:
                st.error(f"{type(e).__name__}: {e}")
            st.stop()

        if answer.explanation:
            st.success(answer.explanation)

        if answer.ok:
            if answer.result.rows:
                st.dataframe(
                    [dict(zip(answer.result.columns, row)) for row in answer.result.rows],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.info("The query ran and matched no rows.")
        else:
            st.error(answer.result.error)

        st.markdown("##### SQL")
        st.code(answer.sql, language="sql")

        a, b, c = st.columns(3)
        a.metric("LLM calls", answer.calls)
        b.metric("Agents", len(answer.agents_used))
        c.metric("Attempts", answer.attempt_count)

        # The repair story is the most interesting thing the UI can show, so it
        # gets its own section rather than being buried in the trace.
        if answer.was_repaired:
            st.markdown("##### What the repair loop did")
            for i, attempt in enumerate(answer.attempts):
                label = {"accepted": "accepted", "repairing": "rejected",
                         "exhausted": "gave up", "no-change": "no change"}.get(
                    attempt.action, attempt.action)
                st.markdown(f"**Attempt {i + 1}** — {label}")
                st.code(attempt.sql, language="sql")
                if not attempt.result.ok:
                    st.caption(f"database said: `{attempt.result.error}`")

    st.caption(
        f"`{strategy}` · repair `{repair.value}` · {answer.usage.summary()}"
    )
