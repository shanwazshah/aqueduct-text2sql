# Architecture

How a question becomes an answer, and where each decision is made.

Two things in this diagram are the project's findings rather than its design, and
both are noted where they appear: the fork after execution (step 10), and the
fact that the Router is mechanical rather than a model.

```
┌───────────────────────────────────────────────────────────────────┐
│ 1. ENTRY LAYER                                                    │
│    Streamlit UI / command-line interface                          │
└────────────────────────────────┬──────────────────────────────────┘
                                 │ natural-language question
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│ 2. CREW ORCHESTRATOR                                              │
│    Controls generation, routing, execution, review and repair     │
└────────────────┬─────────────────────────────────┬────────────────┘
                 │                                 │
                 ▼                                 ▼
┌────────────────────────────────┐  ┌───────────────────────────────┐
│ 3. SCHEMA INTROSPECTION        │  │ 4. ERROR MEMORY               │
│                                │  │                               │
│ • Tables and columns           │  │ • Verified repair lessons     │
│ • Types and primary keys       │  │ • Retrieved by question       │
│ • Foreign keys                 │  │ • Scoped by database          │
│ • Row counts and sample values │  │                               │
└────────────────┬───────────────┘  └───────────────┬───────────────┘
                 │                                  │
                 └────────────────┬─────────────────┘
                                  │ context
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 5. GENERATION STRATEGY                                            │
│                                                                   │
│  1. Direct                 6. Orchestrator                        │
│  2. ReAct                  7. Deep                                │
│  3. Chain                  8. Deep-seeded                         │
│  4. Parallel               9. Self-consistency                    │
│  5. Evaluator-optimizer                                           │
│                                                                   │
│ Common interface: generate(context) -> SQL draft                  │
│ Some strategies inspect and test the database through tools.      │
└────────────────────────────────┬──────────────────────────────────┘
                                 │ uses
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│ 6. LLM LAYER                                                      │
│                                                                   │
│ OpenAI-compatible client / structured output / response cache     │
│ Used by strategies, Critic, Fixer and Analyst.                    │
│ NOT used by the Router - see below.                               │
└────────────────────────────────┬──────────────────────────────────┘
                                 │ generated SQL draft
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│ 7. ROUTER  (optional)                                             │
│                                                                   │
│ Reads the parsed SQL and chooses how much verification to spend:  │
│                                                                   │
│   TRUST  -> execution repair only                                 │
│   VERIFY -> execution repair + Critic                             │
│                                                                   │
│ Mechanical: joins, subqueries, HAVING and arithmetic are read     │
│ from the AST at zero cost. A model router exists only as a        │
│ comparison - it would spend the call the Router exists to save.   │
│                                                                   │
│ It does not select the generation strategy. Measured: nothing     │
│ generates better than one call, so there is nothing to route to.  │
└────────────────────────────────┬──────────────────────────────────┘
                                 │ SQL + repair mode
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│ 8. SQL SAFETY LAYER                                               │
│                                                                   │
│ Parse to AST / require exactly one read statement / reject any    │
│ write or admin node at any depth / inject or tighten LIMIT        │
└────────────────────────────────┬──────────────────────────────────┘
                                 │ safe SQL
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│ 9. DATABASE EXECUTION                                             │
│                                                                   │
│ Execute the query and return rows, or a readable database error   │
└────────────────┬─────────────────────────────────┬────────────────┘
                 │ failed                          │ succeeded
                 ▼                                 ▼
┌────────────────────────────────┐  ┌───────────────────────────────┐
│ 10A. FAILURE PATH              │  │ 10B. SUCCESS PATH             │
│                                │  │                               │
│ Execution repair enabled?      │  │ Critique enabled?             │
│                                │  │                               │
│ YES -> database error +        │  │ YES -> Critic reviews the SQL │
│        schema hints -> Fixer   │  │         ├─ clean  -> accept   │
│                                │  │         └─ issues -> Fixer    │
│ NO  -> return failed answer    │  │                               │
│                                │  │ NO  -> accept immediately     │
│ The Critic is skipped here on  │  │                               │
│ purpose: the database has      │  │ The only question left is     │
│ already said precisely what    │  │ whether a query that RAN is   │
│ is wrong, and a model would    │  │ also RIGHT - which is the     │
│ only say it more vaguely.      │  │ one a model can speak to.     │
└────────────────┬───────────────┘  └───────────────┬───────────────┘
                 │                                  │
                 └────────────────┬─────────────────┘
                                  │ repair required
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│ 11. FIXER                                                         │
│                                                                   │
│ Question + failing SQL + schema + feedback -> corrected SQL       │
└────────────────────────────────┬──────────────────────────────────┘
                                 │ corrected SQL
                                 └─────────────► back to step 8


                          ACCEPTED RESULT
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────┐
│ 12. FINAL ANSWER                                                  │
│                                                                   │
│ • Final SQL              • Agents used and model calls            │
│ • Rows and columns       • Optional English explanation           │
│ • Attempt history        • Success or failure status              │
└───────────────────────────────────────────────────────────────────┘


                     SUPPORTING THE COMPLETE FLOW

┌────────────────────────────────┐  ┌───────────────────────────────┐
│ 13. OBSERVABILITY              │  │ 14. CONFIGURATION             │
│                                │  │                               │
│ • Span tree of agents          │  │ • Models, per agent role      │
│ • Time and model calls         │  │ • Endpoint (base_url)         │
│ • Decisions and errors         │  │ • Database URL                │
│                                │  │ • Repair mode, cache, limits  │
└────────────────────────────────┘  └───────────────────────────────┘


                   SEPARATE OFFLINE EVALUATION SYSTEM

┌───────────────────────────────────────────────────────────────────┐
│ 15. EVALUATION                                                    │
│                                                                   │
│         question set -> Crew -> predicted SQL                     │
│                                      +                            │
│                                 reference SQL                     │
│                                      │                            │
│                                      ▼                            │
│                             execute both queries                  │
│                                      │                            │
│                                      ▼                            │
│                            compare result sets                    │
│                                      │                            │
│                                      ▼                            │
│  generation EX / final EX / calls / latency / BIRD comparisons    │
└───────────────────────────────────────────────────────────────────┘
```

---

## The two boxes that are findings, not design

**Step 10 — why the fork is shaped that way.** A failed query carries a precise,
free, always-correct explanation of what is wrong: the database said so. Asking a
model to restate that costs a call and loses precision. So failure goes straight
to the Fixer, and the Critic is reserved for the only case where a model has
anything to add — a query that *ran* but may still be *wrong*.

Measured: execution feedback was worth up to **+10 points** at near-zero cost;
model critique, **+0.0** for double the calls, measured twice.

**Step 7 — why the Router is not a model.** Query complexity is readable from the
parsed AST: joins, subqueries, `HAVING`, arithmetic, self-joins. A model asked to
rate complexity would spend exactly the call the Router exists to avoid. An
`LLMRouter` is implemented alongside so the claim could be tested rather than
asserted.

It also does not choose the generation strategy, which is what routing usually
means. That was the original design, and the benchmark removed the reason for it:
nothing generates better than a single call, so there is nothing to route *to*.

---

## Where each box lives

| box | module |
|---|---|
| 1 Entry | `ui/app.py` · `src/aqueduct/cli.py` |
| 2 Crew | `src/aqueduct/crew.py` |
| 3 Schema introspection | `src/aqueduct/db/introspect.py` |
| 4 Error memory | `src/aqueduct/agents/memory.py` |
| 5 Strategies | `src/aqueduct/strategies/` |
| 6 LLM layer | `src/aqueduct/llm/client.py` |
| 7 Router | `src/aqueduct/router.py` |
| 8 Safety layer | `src/aqueduct/db/safety.py` |
| 9 Execution | `src/aqueduct/db/engine.py` |
| 10B Critic | `src/aqueduct/agents/critic.py` |
| 11 Fixer | `src/aqueduct/agents/fixer.py` |
| 13 Observability | `src/aqueduct/observability/trace.py` |
| 14 Configuration | `src/aqueduct/config.py` |
| 15 Evaluation | `src/aqueduct/eval/` |
