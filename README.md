# Aqueduct

**Six agentic Text-to-SQL architectures, benchmarked against each other — and the
control run that showed my own headline result was mostly an artifact of my test
set.**

The project started as an implementation exercise from two course notebooks on
agent design. It became an evaluation project, because once the six architectures
were measurable it turned out most of them do not work, and the reason is more
interesting than the code.

```bash
python -m aqueduct.cli ask "Which product category made the most revenue?" --explain
```

*(That needs the install and a local model first — see [Running it](#running-it).)*

---

## The result

100 questions from [BIRD mini-dev](https://bird-bench.github.io/), 11 real
databases, graded by execution accuracy — the generated query is run and its
result set compared against the reference.

| strategy | LLM calls | 3B gen EX | 7B gen EX |
|---|---|---|---|
| **`direct`** — one call | **1** | **29.0%** | **41.0%** |
| `chain` — 5-stage pipeline | ~5 | 20.0% | 35.0% |
| `orchestrator` — planner + specialists | ~6 | 16.0% | 32.0% |

These three are the strategies that were run on BIRD. `react`, `parallel` and
`eval_optimize` were not, and no number is borrowed for them here — mixing a
demo-set score into a BIRD table is the mistake this project already retracted
once, and it is not worth repeating for a tidier table. Their demo-set results
are in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md); the short version is that
`react` generated one correct query in twenty-two at 3B, making no tool call at
all on 14 of them, because a 3B model cannot hold a multi-step plan.

**The single-call baseline wins at both model sizes, using five times less
compute.** Every architecture that replaces one-shot generation with a pipeline
does worse, and the more stages it has, the worse it does.

The mechanism is the same in each case: **every stage inherits the previous
stage's errors and has no way to detect them.** The orchestrator's synthesiser is
instructed to follow its specialists' findings, so a wrong join key from one
worker is faithfully written into the final query.

41% on BIRD mini-dev is a credible 7B-class number — published results for models
this size sit in the 25–45% band.

---

## The retraction

An earlier version of this README reported the gap between `direct` and the
decomposed strategies as **41 points**, measured on a 22-question demo set.

That number was inflated roughly 5× by the test set.

| gap behind `direct` | 3B, demo set | 3B, BIRD | 7B, BIRD |
|---|---|---|---|
| `chain` | **40.9** | **9.0** | **6.0** |
| `orchestrator` | **50.0** | **13.0** | **9.0** |

Holding the model fixed and changing only the benchmark takes the gap from 40.9
to 9.0. Changing the model then takes it from 9.0 to 6.0. **About 90% of the
effect I originally attributed to model capability was my own easy test set.**

In hindsight the mechanism is obvious: `direct` scored 90.9% on the demo set,
leaving 41 points of room beneath it for a gap to occupy. At 29% there is not.

The real conclusions are narrower and better supported:

- the decomposition penalty on a real benchmark is **6–13 points**, not 41–50;
- model scale narrows it by **3–4 points** — small, but consistent in sign across
  two independent pipelines;
- `direct` still wins at both sizes.

Full history in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md), including the entry
that made the original claim.

---

## What actually earned its keep

Nothing that worked was an agent.

**Execution feedback.** Run the query, read the database's error, rewrite. Costs
nothing when the query succeeds, one call when it fails.

| strategy | repair gain, 3B | repair gain, 7B |
|---|---|---|
| `direct` | +4.0 | +1.0 |
| `chain` | **+10.0** | **+6.0** |
| `orchestrator` | +2.0 | +3.0 |

It helps most exactly where generation is weakest — a pipeline emits more
*executable but wrong* SQL, and execution feedback is the signal that catches it.
This held in every phase, on both benchmarks, at both model sizes.

**Model self-critique, by contrast, bought +0.0 for double the calls**, measured
twice. Asked to review `SELECT dept FROM employees` having been told the column is
`department`, a 3B model returned `schema_ok: true, confidence: 0.9`.

**The schema card.** Foreign keys listed explicitly, plus sample values for
categorical columns so the model can see that `status` holds `'shipped'` rather
than guessing `'Shipped'` and silently returning zero rows.

**Doing in code what does not need a model.** Column existence is a
set-membership test. It runs in Python.

---

## Design decisions worth knowing about

**Safety is enforced by a parser, not a prompt.** Generated SQL is parsed to an
AST and rejected unless it is exactly one read statement with no write or admin
node anywhere in the tree, then a row cap is injected. The source notebooks write
*"NEVER run DELETE, DROP"* into the agent persona — a request, not a control.
`tests/test_safety.py` holds 16 adversarial cases including a `DELETE` hidden
inside a CTE and a comment-breakout attempt. All 500 BIRD reference queries pass
the guard with zero false positives.

**The mechanical schema check is parsed, not pattern-matched.** A regex version
flagged output aliases, string literals, type names and function names as
nonexistent columns — every one on a *correct* query, and each would have
triggered a pointless repair.

**The evaluation code is tested adversarially too.** The grader had two false-pass
bugs before it had a single user: both attempts to make column order irrelevant
scored `(min=5, max=10)` and `(min=10, max=5)` as identical. It now compares
positionally, exactly as BIRD's and Spider's official scripts do.

**Generation is measured separately from rescue.** Each result records the
strategy's raw draft graded *before* the repair layer touches it. Without that
column, `react` scored 81.8% while generating one correct query in 22 — the
repair layer was writing every query and the leaderboard was labelling it
`react`.

Full reasoning, including approaches that were tried and dropped:
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## What I would tell you in review

Six instrumentation bugs were found over this project. Zero bugs were found in
the agent logic.

**The measuring apparatus was consistently less reliable than the thing being
measured** — and every one of those bugs flattered the result. None was caught by
reading the code more carefully; each was caught by measuring the same thing a
second way.

Two numbers reached a written conclusion before being caught. The first was
`react`'s original 95.5%: before the tool-call bug was found it generated nothing
at all, and the repair layer wrote every query under its name. The second was the
41-point gap above. Both looked entirely plausible.

---

## Running it

```bash
pip install -e ".[ui,dev]"
python -m aqueduct.cli seed          # build the demo database
python -m aqueduct.cli doctor        # check database + LLM backend
```

Needs any OpenAI-compatible endpoint. Locally that is [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5-coder:3b
```

```bash
python -m aqueduct.cli ask "Which department has the highest total salary spend?"
```

```bash
python -m aqueduct.eval.compare      # strategy leaderboard on the demo set
python -m aqueduct.eval.ablation     # repair signals, head to head
python -m aqueduct.eval.routing      # verification tiers
```

The BIRD sweep runs on Kaggle — import
[`kaggle/aqueduct_bird_kaggle.ipynb`](kaggle/aqueduct_bird_kaggle.ipynb), set
**GPU T4 x2**, **Internet On**, **Persistence: Files only**, and Run All.

---

## Two tiers, one codebase

| Tier | Hardware | Model | For |
|---|---|---|---|
| **Dev** | laptop, 4 GB VRAM | `qwen2.5-coder:3b` | iteration, tests |
| **Eval** | Kaggle 2× T4 | `qwen2.5-coder:7b` | benchmark sweeps |

Both speak the OpenAI protocol, so the backend is a `base_url` in config — nothing
under `agents/` or `strategies/` knows which tier it is on. That is what makes
model size a measurable axis rather than a rewrite, and it is why the control run
above was cheap enough to bother with.

---

## Layout

```
src/aqueduct/
├── db/           engine · introspect · safety (sqlglot AST guard) · seed
├── llm/          one OpenAI-compatible client · disk cache · bounded types
├── agents/       writer · critic · fixer · memory
├── strategies/   direct · react · chain · parallel · eval_optimize · orchestrator
├── router.py     verification tiering, decided from the parsed SQL
├── eval/         BIRD loader · execution-accuracy grader · sweeps · leaderboards
└── observability/ span tree behind the traces and the cost accounting
```

188 tests. `pytest`.

---

## Open question

On **challenging** BIRD questions at 7B, `chain` beat `direct` — **30% vs 25%**.
That is decomposition behaving as intended: hard problem, model capable enough for
the extra structure to pay.

It is also 6 questions against 5, out of 20. A hypothesis, not a finding. The full
mini-dev has 102 challenging questions, which is the sample size that would settle
it.

Configuration lives in `.env` — see [`.env.example`](.env.example).
