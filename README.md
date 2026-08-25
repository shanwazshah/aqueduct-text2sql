# Aqueduct

**An agentic Text-to-SQL crew that sizes itself to the question.**

Ask a question in English. A Lead agent reads it, decides how many specialists the
job needs, and spins up only those. A simple count gets two agents and an answer
in seconds. A four-table analytical question gets seven.

The other half of the project is the part that makes the first half a claim rather
than a hope: an evaluation harness that measures whether the extra agents are
worth what they cost.

```bash
python -m aqueduct.cli ask "Which product category made the most revenue?" --explain
```

---

## Why this exists

This started from two teaching notebooks on agent design. Between them they build
six different ways to turn a question into SQL — a ReAct tool-using agent, prompt
chaining, parallel critics, an evaluator-optimizer loop, and an orchestrator with
specialist workers.

They leave two things undone, and this project is those two things:

**1. Routing is never applied to Text-to-SQL.** The routing lesson routes customer
support tickets. Every other pattern gets a SQL implementation; that one does not.
Yet routing is exactly what a real system needs — `SELECT COUNT(*)` should not pay
for a seven-call pipeline.

**2. Nothing is measured.** Six strategies, no evidence about which is better, at
what cost, on which kind of question. So every design choice downstream is taste.

---

## The crew

| Agent | Job | Spun up when |
|---|---|---|
| **Lead** | Reads the question, decides the headcount | always |
| **Scout** | Finds which tables actually matter | wide schemas |
| **Writer** | Writes the SQL | always |
| **Specialists** | Joins, aggregation, filters | the query is hard enough to need them |
| **Critics** | Review the query before it runs; vote | when correctness matters more than latency |
| **Runner** | Executes it, safely | always |
| **Fixer** | Rewrites using the reason it failed | something went wrong |
| **Analyst** | Turns rows into an answer | the caller wants prose |

Underneath sits an **error memory**: verified corrections persist, so a mistake
made once is less likely to be made again.

---

## Design decisions worth knowing about

**Safety is enforced by a parser, not a prompt.** Generated SQL is parsed into an
AST and rejected unless it is exactly one read statement, with no write or admin
node anywhere in the tree. The source notebooks instead write *"NEVER run DELETE,
DROP"* into the agent persona — a request a 3B model will eventually ignore. There
are 16 adversarial cases in `tests/test_safety.py`, including a `DELETE` hidden
inside a CTE.

**Checks that can be mechanical are mechanical.** Column existence is a
set-membership test, so it runs in Python, not through a model. The notebooks ask
the LLM for `schema_ok`; measured here, a 3B model shown a hallucinated column
returned `schema_ok: true, confidence: 0.9`. The model is asked only what is
genuinely semantic — wrong aggregate, wrong join, missing filter.

**Feedback is ranked by reliability.** The database's `no such column: dept` is
ground truth, costs nothing, and arrives before any model is consulted. Model
critique is the secondary signal, for queries that run cleanly and answer the
wrong question.

**The evaluation code is tested adversarially too.** The grader had two false-pass
bugs before it had a single user — both attempts to make column order irrelevant
scored genuinely different answers as equal. A metric that flatters itself
corrupts every number downstream of it.

Full reasoning, including the approaches that failed: [`docs/DECISIONS.md`](docs/DECISIONS.md).
Dated results with before-and-after numbers: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## Running it

```bash
pip install -e ".[ui,dev]"
python -m aqueduct.cli seed          # build the demo database
python -m aqueduct.cli doctor        # check database + LLM backend
```

Needs an OpenAI-compatible endpoint. Locally that is [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5-coder:3b
```

Then:

```bash
python -m aqueduct.cli ask "Which department has the highest total salary spend?"
```

```bash
python -m aqueduct.eval.runner       # score the crew on the demo question set
```

```bash
python -m aqueduct.eval.ablation     # compare repair strategies head to head
```

---

## Two tiers, one codebase

| Tier | Hardware | Model | For |
|---|---|---|---|
| **Dev** | laptop, 4 GB VRAM | `qwen2.5-coder:3b` via Ollama | iteration, tests, UI |
| **Eval** | Kaggle 2× T4 | `qwen2.5-coder:7b`+ via vLLM | benchmark sweeps |

Ollama and vLLM both speak the OpenAI protocol, so the backend is a `base_url` in
config. Nothing in `agents/` knows which tier it is on — which makes model size a
measurable axis rather than a rewrite.

---

## Status

| Phase | | |
|---|---|---|
| 0 | Foundation — config, safety guard, demo database | done |
| 1 | Writer + Runner, baseline measured | done |
| 2 | Critics, Fixer, error memory | done |
| 3 | Four generation strategies behind one interface | next |
| 4 | The Lead — routing to strategy and model | |
| 5 | Schema linking for wide databases | |
| 6 | BIRD benchmark on Kaggle | |
| 7 | FastAPI + Streamlit | |

Configuration lives in `.env` — see [`.env.example`](.env.example).
