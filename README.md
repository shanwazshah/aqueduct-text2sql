# Aqueduct

**Nine ways to turn a question into SQL, benchmarked against each other.**
The simplest one wins — and two of my own headline numbers turned out to be
measurement bugs before I caught them.

```bash
python -m aqueduct.cli ask "Which product category made the most revenue?" --explain
```

---

## The headline

Multi-agent architectures are the standard advice for tasks like this. Measured
on [BIRD](https://bird-bench.github.io/) — a real benchmark, 11 databases, graded
by running the SQL and comparing results — they lose to a single well-written
prompt.

**100 questions, `qwen2.5-coder` at two sizes:**

| strategy | LLM calls | 3B | 7B |
|---|---|---|---|
| **`direct`** — one call | **1** | **29.0%** | **41.0%** |
| `chain` — 5-stage pipeline | ~5 | 20.0% | 35.0% |
| `orchestrator` — planner + 5 specialists | ~6 | 16.0% | 32.0% |

**Why:** every stage inherits the previous stage's errors and has no way to
detect them. The orchestrator's synthesiser is *instructed* to follow its
specialists, so one worker's wrong join key goes straight into the final query.

*(41% is a credible 7B-class BIRD score — published results sit at 25–45%.)*

---

## What finally beat it wasn't an architecture

A later phase built a proper **deep agent**: it plans, inspects tables, checks
what a column actually contains, and tests queries before committing.

It was measured against a control most people skip — spend the same budget just
**sampling the baseline prompt five times and voting**.

**50 challenging BIRD questions, 7B:**

| arm | accuracy | calls | **seconds/question** |
|---|---|---|---|
| `direct` | 20.0% | 1.4 | **6** |
| `self_consistency` — 5 samples, voted | **28.0%** | 5.1 | **19** |
| `deep_seeded` — the deep agent | **28.0%** | 9.8 | **318** |

Identical accuracy. The agent took **16× longer**.

The four possible outcomes were written into
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) **before** the run. This was outcome
2: *"the gain was the budget, not the design. Report it that way."*
Per-question results:
[`data/bird/`](data/bird/results_phase9_lite_7b_challenging.json).

One thing did improve with scale: at 3B the agent decided it was finished on 1
question in 10; at 7B, on 27 of 50. **The scaffolding started working and it
bought nothing** — a sharper result than it failing would have been.

---

## What actually helped

Nothing that worked was an agent.

**Execution feedback** — run the query, read the database's error, rewrite. Free
when the query succeeds, one call when it fails.

| strategy | gain, 3B | gain, 7B |
|---|---|---|
| `direct` | +4.0 | +1.0 |
| `chain` | **+10.0** | **+6.0** |
| `orchestrator` | +2.0 | +3.0 |

It helps most where generation is weakest — pipelines produce more
*executable but wrong* SQL, and the database is what catches it.

**Asking the model to review its own work** was worth **+0.0 for double the
calls**, measured twice. Shown `SELECT dept FROM employees` and told the column
is `department`, a 3B model replied `schema_ok: true, confidence: 0.9`.

**The schema card** — foreign keys listed explicitly, plus sample values, so the
model can see `status` holds `'shipped'` instead of guessing `'Shipped'` and
silently returning nothing.

**Doing in code what doesn't need a model** — column existence is a lookup, so it
runs in Python.

---

## Two numbers I got wrong

Both looked completely plausible. Neither was caught by re-reading code — only by
measuring the same thing a second way.

**A strategy scored 95.5% while generating nothing.** `react` made no tool calls
at all on 14 of 22 questions; the repair layer wrote every query and the
leaderboard credited the agent. The tell wasn't the score — it was a repair count
of 22/22 against `direct`'s 1. Results now record the raw draft graded *before*
repair touches it.

**A "41-point gap" was 90% my own test set.** I reported that decomposition cost
41 points, measured on 22 easy questions. The control run:

| gap behind `direct` | 22-question demo | BIRD, same model | BIRD, bigger model |
|---|---|---|---|
| `chain` | **40.9** | **9.0** | **6.0** |

Changing only the benchmark took it from 40.9 to 9.0. The real penalty is
**6–13 points**, not 41–50. `direct` scored 90.9% on the easy set, leaving 41
points of room beneath it for a gap to occupy; at 29% there isn't.

**Eight instrumentation bugs found. Zero in the agent logic.** Seven flattered the
result. The measuring apparatus was consistently less trustworthy than the thing
it measured.

---

## Engineering worth noting

**Safety is enforced by a parser, not a prompt.** Every generated query is parsed
to an AST and rejected unless it's exactly one read statement with no write
anywhere in the tree. `tests/test_safety.py` holds 16 attacks including a
`DELETE` hidden inside a CTE. All 500 BIRD reference queries pass with zero false
positives.

**The grader is tested adversarially too.** It had three false-pass bugs — two
before it had a single user, one that survived to Phase 7 where a prediction could
dodge the row-order check by not sorting.

**Results are checkpointed and resumable.** A benchmark sweep is hours of GPU; a
dropped session resumes rather than restarts.

Full reasoning, including rejected approaches:
[`docs/DECISIONS.md`](docs/DECISIONS.md) ·
Dated results including the retraction: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)

---

## Running it

```bash
pip install -e ".[ui,dev]"
python -m aqueduct.cli seed      # build the demo database
python -m aqueduct.cli doctor    # check database + model are reachable
```

Needs any OpenAI-compatible endpoint. Locally that's [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5-coder:3b
python -m aqueduct.cli ask "Which department has the highest total salary spend?"
```

**The UI** — watch agents appear live, with their cost:

```bash
streamlit run ui/app.py
```

**The experiments:**

```bash
python -m aqueduct.eval.compare      # strategy leaderboard
python -m aqueduct.eval.ablation     # which repair signal works
python -m aqueduct.eval.routing      # how much verification is worth
```

BIRD runs on a GPU — import a notebook from [`kaggle/`](kaggle/), set **GPU T4
x2**, **Internet On**, **Persistence: Files only**, Run All.

---

## Layout

```
src/aqueduct/
├── db/            engine · schema introspection · sqlglot safety guard
├── llm/           one OpenAI-compatible client (Ollama, vLLM, OpenAI) · cache
├── agents/        writer · critic · fixer · error memory
├── strategies/    direct · react · chain · parallel · eval_optimize
│                  orchestrator · deep · deep_seeded · self_consistency
├── router.py      how much verification a query deserves, from its parsed AST
├── eval/          BIRD loader · execution-accuracy grader · sweeps · recovery
└── observability/ span tree behind the live trace and cost accounting
```

**236 tests** — `pytest`. Config in `.env`, see [`.env.example`](.env.example).

The same code runs against a 3B model on a laptop and a 7B on a Kaggle T4 — the
backend is a `base_url`, so model size is a measurable axis rather than a rewrite.

---

## Open question

On the hardest BIRD questions, `chain` beat `direct` — **30% vs 25%**. That's
decomposition working as intended: hard problem, capable enough model.

It's also 6 questions against 5, out of 20. **A hypothesis, not a finding.** The
full stratum has 102 questions, which is the sample size that would settle it.
