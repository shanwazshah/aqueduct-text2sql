# Aqueduct — An Agentic Text-to-SQL System

> **Historical — the plan as written on 2026-08-25, before any measurement.**
>
> Kept unedited, because the distance between it and the outcome is part of the
> record. The router described here as "the spine" was built, measured, and found
> unnecessary. The multi-agent architectures it treats as the deliverable lose to
> a single LLM call. The 41-point gap it was written to explain turned out to be
> an artifact of the test set.
>
> For what the project actually found:
> [README.md](README.md) · [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)

Inspired by *Agents: Foundations & Planning* (class 1) and *Advanced Agent
Concepts* (class 2). Those notebooks build six ways to generate SQL. This project
set out to decide **which one to use, when** — and to prove the decision with
numbers.

---

## 1. The thesis

The notebooks leave two things undone:

1. **Routing was never applied to Text2SQL.** Workflow 2 in class 2 routes *customer support tickets*.
   Every other workflow got a Text2SQL implementation; routing did not. Yet routing is precisely what
   a real system needs — a `COUNT(*)` question should not pay for a 7-call orchestrator-worker pipeline.
2. **Nothing is measured.** Six strategies exist, and no evidence says which is better, at what cost,
   at what latency, on which kind of question.

Aqueduct closes both. The router is the spine; the evaluation harness is the proof.

---

## 2. Execution model — two tiers, one codebase

| Tier | Where | Hardware | Model | Purpose |
|---|---|---|---|---|
| **Dev** | This laptop | i5-8300H · 16 GB · GTX 1050 Ti (4 GB) | `qwen2.5-coder:3b` via Ollama | Fast iteration, plumbing correctness, unit tests, UI work |
| **Eval** | Kaggle | 2× Tesla T4 (16 GB each) | `qwen2.5-coder:7b`+ via vLLM | Benchmark sweeps that produce the leaderboard |

The two tiers run **identical agent code**. Both Ollama and vLLM expose an OpenAI-compatible
`/v1` endpoint, so the backend is a `base_url` + model name in config — nothing in `strategies/`
or `agent/` knows which tier it is on.

This is not a workaround; it is the project's third result axis. Alongside *strategy* and
*question difficulty* we get **model size** — the quality/cost curve from 3B to 7B to 14B.

**Kaggle budget reality:** 30 GPU-hours/week, 12-hour session cap, ephemeral disk. Therefore the
harness is **resumable** (checkpoint after every question), **cached** (keyed on
`sha256(model + prompt)`), and **subset-first** (stratified 200 for the main sweep, full 500 only
for the top two strategies).

---

## 3. Repository structure

```
Text-to-SQL/
├── README.md
├── PROJECT_PLAN.md                  ← this file
├── pyproject.toml
├── .env.example
│
├── data/
│   ├── raw/                         BIRD mini-dev download (gitignored)
│   ├── db/                          SQLite databases
│   └── demo/                        seed demo DB (the notebooks' employees table, extended)
│
├── src/t2s/
│   ├── config.py                    pydantic-settings; model registry per role
│   │
│   ├── llm/
│   │   ├── base.py                  LLMClient protocol: chat(), structured(), with_tools()
│   │   ├── openai_compat.py         ONE client — serves Ollama, vLLM, and cloud
│   │   ├── structured.py            Pydantic model → JSON schema → guaranteed-valid output
│   │   ├── registry.py              role → model  (router=3b, sql=7b, critic=3b)
│   │   └── cache.py                 disk cache; makes eval re-runs free
│   │
│   ├── db/
│   │   ├── engine.py                SQLAlchemy factory, dialect-aware (SQLite → MySQL)
│   │   ├── introspect.py            list_tables · get_table_schema · sample_rows · fk_graph
│   │   ├── schema_card.py           compact schema serialization (M-schema style)
│   │   └── safety.py                sqlglot AST guard: SELECT-only, LIMIT cap, timeout
│   │
│   ├── agent/                       ── CLASS 1 ──
│   │   ├── persona.py               Component 1: persona dict → system prompt
│   │   ├── planner.py               Component 2: LLM as planner
│   │   ├── tools.py                 Component 4: tool fns + JSON schemas + dispatch map
│   │   ├── memory.py                Component 5: conversation + persistent error memory
│   │   └── react.py                 the ReAct loop (`run_agent`, hardened)
│   │
│   ├── selfimprove/                 ── CLASS 2, sections 8–11 ──
│   │   ├── critique.py              structured self-evaluation
│   │   ├── refine.py                feedback-conditioned repair
│   │   └── loop.py                  SelfImprovingSQLAgent + execution-aware feedback
│   │
│   ├── strategies/                  ── CLASS 2, workflows 1/3/4/5 ──
│   │   ├── base.py                  Strategy protocol → SQLResult
│   │   ├── direct.py                single-shot baseline (the control group)
│   │   ├── react_agent.py           class 1's agent
│   │   ├── chain.py                 W1: intent → ground → generate → verify → repair
│   │   ├── parallel.py              W3: 3 voting critics + safety sectioning
│   │   ├── eval_optimize.py         W4: generator ⇄ evaluator loop
│   │   └── orchestrator.py          W5: planner + 5 specialist workers + synthesis
│   │
│   ├── router.py                    ── W2, THE GAP ── complexity → strategy + model
│   │
│   ├── retrieval/
│   │   ├── index.py                 embed table/column descriptions (nomic-embed-text)
│   │   └── linker.py                top-k table pruning — required for BIRD's wide schemas
│   │
│   ├── observability/
│   │   ├── trace.py                 span tree; token, cost, latency accounting
│   │   └── render.py                CLI trace rendering
│   │
│   ├── eval/
│   │   ├── bird.py                  BIRD mini-dev loader + difficulty stratification
│   │   ├── metrics.py               Execution Accuracy, VES, soft-F1
│   │   ├── runner.py                resumable, cached, concurrent sweep
│   │   └── report.py                markdown + HTML leaderboard
│   │
│   ├── api/main.py                  FastAPI /ask with SSE trace streaming
│   └── cli.py                       typer: t2s ask · eval · setup-db · serve
│
├── ui/app.py                        Streamlit: chat + live trace + cross-strategy SQL diff
├── kaggle/
│   ├── 00_bootstrap.ipynb           install vLLM, serve model, run sweep, save results
│   └── README.md                    how to reproduce the leaderboard
├── notebooks/
│   ├── 01_foundations_walkthrough.ipynb
│   └── 02_workflow_comparison.ipynb   ← the money chart
└── tests/
```

---

## 4. Build order

Each phase ends with something that **runs**.

| # | Phase | Delivers | Tier |
|---|---|---|---|
| 0 | Scaffold | config, SQLite demo DB, `safety.py` guard, LLM client + structured output | Dev |
| 1 | Foundations | persona · tools · planner · ReAct loop → agent answers questions end-to-end | Dev |
| 2 | Self-improvement | critique → execute → refine, persistent error memory | Dev |
| 3 | Strategies | four workflows behind one `Strategy` interface | Dev |
| 4 | Router | complexity classifier → strategy **and** model selection | Dev |
| 5 | Schema linking | top-k table retrieval — without this, BIRD's wide schemas blow context | Dev |
| 6 | **Evaluation** | BIRD sweep, leaderboard, the comparison chart | **Kaggle** |
| 7 | Surface | FastAPI + Streamlit + tracing + README | Dev |

---

## 5. Three upgrades over the notebooks

**a. Structured output replaces fence-stripping.**
Class 2 parses JSON like this:

```python
raw.removeprefix("```json").removesuffix("```")
try: json.loads(clean)
except JSONDecodeError: return {"error": ...}
```

Against a 3B local model that fallback path fires constantly. Ollama's `format` parameter and
vLLM's `guided_json` both accept a JSON Schema and **constrain decoding** — invalid JSON becomes
unrepresentable. Every structured call in the project goes through a Pydantic model. Verified
working on this machine against Ollama 0.32.15.

**b. The router picks a model, not just a strategy.**
Local inference makes model choice a live decision per call: 3B for classification and critique,
7B for SQL generation. Cloud APIs hide this; running our own serving exposes it.

**c. Safety is enforced by a parser, not a prompt.**
The notebooks' persona *asks* the model not to write `DROP`. Aqueduct parses generated SQL with
`sqlglot` and rejects anything that is not a single read-only `SELECT`, then caps `LIMIT` and
applies a statement timeout. A rule the model cannot violate beats a rule it is asked to respect.

---

## 6. The target result

```
Strategy              EX%    VES    calls/q   p50 latency   notes
────────────────────────────────────────────────────────────────────
direct                 —      —        1.0         —        control
react_agent            —      —        4.2         —        class 1
chain                  —      —        5.8         —        W1
parallel               —      —        6.1         —        W3
eval_optimize          —      —        4.4         —        W4
orchestrator           —      —        7.3         —        W5
router  ★              —      —        2.6         —        this project
```

The claim to prove: **the router reaches within ~2 points of the best single strategy at roughly a
third of its cost.** If it does not, that is a finding worth reporting too — and the harness is what
makes either outcome credible.
