# Experiment log

The lab notebook: what was tried, what the numbers were, what changed as a result.
Entries are appended, never rewritten — including the ones that did not work.

Metric definitions live in `src/aqueduct/eval/metrics.py`. The headline number is
**EX (execution accuracy)**: the fraction of questions where the crew's query
returns the same result set as the reference query.

---

## 2026-08-25 — Phase 0: foundation

**Environment.** Dev tier: i5-8300H, 16 GB RAM, GTX 1050 Ti (4 GB VRAM),
`qwen2.5-coder:3b` on Ollama 0.32.15. Eval tier not yet used.

**Why qwen2.5-coder:3b for the dev tier.** At Q4 it is ~1.9 GB, so it fits
entirely in 4 GB of VRAM and runs at usable speed. The 7B variant is ~4.7 GB and
would spill to CPU on this card. `llama3.2:3b` was already installed but is a
general chat model — not code-tuned, and weak at multi-step tool calling.

**Structured output verified.** Ollama's `format` parameter accepts a JSON Schema
and constrains decoding. Confirmed against `llama3.2:3b`: a request for
`{schema_ok, confidence, issues}` returned valid conforming JSON first try, in
21.4 s for 95 tokens.

This matters more than it looks. The notebooks parse structured LLM output like
this:

```python
raw.removeprefix("```json").removesuffix("```")
try: json.loads(clean)
except JSONDecodeError: return {"error": ...}
```

Against a 3B model that fallback path fires often, and every time it does, an
agent silently degrades to a default verdict. Schema-constrained decoding makes
malformed JSON unrepresentable rather than merely unlikely.

**Note on latency.** 21.4 s for 95 tokens is ~4.4 tok/s, and that was on the
smaller model with a cold load. This is the constraint that justifies the
two-tier plan — a 7-agent crew at this speed is roughly a two-minute round trip.
Fine for development, unusable for a 200-question sweep. Real throughput numbers
belong in Phase 1 once the crew is doing actual work.

**Safety guard results.** 16 adversarial inputs, 29 assertions in
`tests/test_safety.py`. All pass.

| Attack | Result |
|---|---|
| `DROP` / `DELETE` / `UPDATE` / `INSERT` / `ALTER` / `CREATE` / `TRUNCATE` | blocked at root check |
| `SELECT 1; DROP TABLE employees` | blocked — statement count |
| `WITH gone AS (DELETE … RETURNING *) SELECT * FROM gone` | blocked — tree walk found the write |
| `PRAGMA` / `ATTACH DATABASE` | blocked — admin nodes |
| `SELECT load_extension('evil.so')` | blocked — function denylist |
| prose instead of SQL | blocked — parse failure |
| legitimate joins, CTEs, unions, subqueries | allowed |

**One finding worth recording.** sqlglot rewrites `--` line comments into `/* */`
blocks on output. I tested whether a payload containing `*/` could close the
comment early and smuggle in a second statement:

```
input:    SELECT * FROM employees -- x */ ; DROP TABLE employees
emitted:  SELECT * FROM employees /* x * / ; DROP TABLE employees */ LIMIT 500
reparsed: 1 statement (Select)
```

sqlglot escapes `*/` as `* /`, so the attack fails. Correct behaviour, but the
mitigation stayed in anyway (see D3) — comments are now stripped entirely and the
emitted string is re-parsed before execution. Depending on a library's escaping
staying correct across versions is a thin margin for a control this important.

**Demo database.** 5 tables, 50 rows. Deliberately includes four traps for small
models (see D5). A three-table join through the guard returned correct results in
3.8 ms; `SELECT dept FROM employees` failed with `no such column: dept` — which is
exactly the shape of feedback the Fixer agent will consume.

**Status.** Foundation done. Nothing agentic yet — no LLM call is made anywhere
in the codebase at this point. Next: the LLM client, then Writer + Runner as the
first working agent.

---

<!-- Next entry: Phase 1 — first agent end-to-end. Record baseline EX before
     adding any self-correction, so later improvements have something to beat. -->

## 2026-08-25 — Phase 1: first working crew (baseline)

**What runs now.** Writer → Runner. A question goes in, the Writer produces SQL
against the introspected schema, the guard checks it, the Runner executes it.
Two agents, no self-correction, no routing. This is the control group (D7).

### Baseline: 91.7% EX (11/12), `qwen2.5-coder:3b`

| Cut | Score |
|---|---|
| **Overall** | **11/12 (91.7%)** |
| easy | 3/3 |
| medium | 5/6 |
| hard | 3/3 |
| queries that failed to run | 0 |

Trap-by-trap, all passed: `amount-vs-price` 2/2, `nullable-fk` 2/2,
`name-collision` 2/2, `self-join` 1/1, `date-as-text` 1/1.

**Read this number with suspicion.** Twelve questions against a five-table
schema is a smoke test, not a benchmark. It is here to catch regressions between
phases and to make the plumbing measurable — the real number comes from BIRD on
the Kaggle tier. A 3B model does not genuinely solve 92% of Text-to-SQL.

**The traps all passed, which was not expected.** The schema card appears to be
doing the work: foreign keys are listed explicitly, and sample values are shown
for categorical columns, so the model can see that `status` holds `'shipped'`
rather than guessing `'Shipped'`. Worth testing directly later by ablating the
sample values and re-running — if EX drops, that quantifies how much the schema
card is worth, which is a better result than the 91.7% itself.

**The one failure (q09)** is a grading edge case rather than bad SQL. Asked which
employees earn above average, the model returned `name, salary`; the reference
returns `name`. The rows are right, there is an extra column. Scored wrong on
purpose — see D6.

### Latency, measured

| | |
|---|---|
| Writer, single question | **18.0 s** (580 tokens) |
| Runner (SQL execution) | 12 ms |
| Full 12-question sweep, cold | 54.6 s |
| Full 12-question sweep, cached | 0.1 s |

The Writer dominates completely — SQL execution is roughly 0.07% of the total.
Everything expensive here is the model, which is the entire justification for the
two-tier plan.

The cache is doing what it was built for: 54.6 s → 0.1 s on re-run. Every
grader change from here on is free to re-evaluate, which is what made iterating
on D6 practical rather than a 55-second wait per attempt.

### Findings

**1. The 3B model is a poor critic.** While verifying structured output, it was
asked to review `SELECT dept FROM employees` having been told the real column is
`department`. It returned `schema_ok: true, confidence: 0.9`. Confidently wrong
about the exact error class the critic agents exist to catch.

This is a real signal about Phase 2 design. Anthropic's Self-Refine pattern (and
the source notebook's `SelfImprovingSQLAgent`) leans on the model to evaluate its
own output. At 3B that evaluation is close to noise. **Execution feedback should
be the primary repair signal and model critique the secondary one** — the
database's `no such column: dept` is ground truth, and free.

Worth measuring explicitly in Phase 2: critique-driven repair vs
execution-driven repair vs both, on the same question set.

**2. Structured output needs no abstraction layer.** Ollama's `/v1` endpoint
accepts standard `response_format: {"type": "json_schema", ...}`. Tested before
building for it; the planned adapter was never needed (see D4).

**3. My own grader had two false-pass bugs before it had any users.** Both
attempts to make column order irrelevant scored genuinely different answers as
equal. Full write-up in D6. The lesson worth keeping: the evaluation code needs
adversarial tests as much as the agent code does, because a flattering metric
corrupts every number downstream of it silently.

**Status.** Baseline established. Next: Phase 2 — critics and the Fixer, measured
against 91.7%.

## 2026-08-25 — Phase 2: repair, and which feedback signal actually works

**What runs now.** Writer → Runner → Critic → Fixer, with a persistent error
memory. The repair loop is switchable (`RepairMode`) so the four configurations
can be compared rather than assumed.

### The experiment

Same 22 questions, same grader, same first draft in every mode (temperature 0 and
a response cache guarantee it), each mode given its own empty memory so lessons
cannot leak between runs. Only the repair signal varies.

| mode | EX | correct | repairs fired | calls/question | vs. baseline |
|---|---|---|---|---|---|
| `none` | 90.9% | 20/22 | 0 | 11.5 | — |
| **`execution`** | **95.5%** | **21/22** | 1 | **11.9** | **+4.5** |
| `critique` | 90.9% | 20/22 | 0 | 22.6 | +0.0 |
| `both` | 95.5% | 21/22 | 1 | 23.4 | +4.5 |

**Execution feedback bought +4.5 points for 3% more calls. Model critique bought
nothing for double the calls.** `both` is identical to `execution` alone — the
critic contributed no repairs on top of it.

The repaired question was h02, *"which employees have never handled an order?"*.
The Writer joined `order_items` on `employee_id`, a column that lives on `orders`.
The database said `no such column: oi.employee_id`, the Fixer corrected the join,
and the answer became right. Total additional cost: one call.

**Honest limits on this result.** The +4.5 points is *one question*. At n=22 with
a single repair event, this establishes a direction, not a magnitude. What makes
the direction credible is the mechanism rather than the sample: an execution error
is ground truth from the database, it is free, and it arrives before any model is
consulted. What would make the magnitude credible is BIRD on the Kaggle tier, and
that number is not in yet.

**The critic found zero issues across 22 queries.** Combined with the Phase 1
observation — a hallucinated column rated `schema_ok: true, confidence: 0.9` — the
3B critic looks too permissive to be worth 11 extra calls per question. Whether a
7B critic earns its cost is a Phase 6 question. It is not a foregone conclusion
that it does.

**This contradicts the source material's design.** The notebooks'
`SelfImprovingSQLAgent` leads with `critique_sql` and treats the database error as
supplementary. Measured on a small model, that ordering is backwards.

### The regression that showed up first, and what caused it

The first run of this ablation produced a *null* result — all four modes at 90.9%,
and worse, `execution` fixed h02 but simultaneously broke h04, which had passed
under `none`. A repair mode causing a regression in an unrelated question needed
explaining.

Isolated it:

| condition | h04 |
|---|---|
| memory disabled | correct |
| one lesson from h02 in memory | **wrong** |

The h02 lesson was retrieved for h04 because both questions mention orders and
employees. Its corrected query used `LEFT JOIN orders ... WHERE ... IS NULL`, and
the Writer reproduced that join shape on h04, adding a `WHERE o.status = 'shipped'`
filter that nothing had asked for. A correct answer became a wrong one.

Two faults, both fixed and both now regression-tested (see D9):

1. **Retrieval admitted any lesson sharing one keyword.** Now Jaccard overlap with
   a 0.25 floor.
2. **Lessons were rendered as full SQL.** Showing a corrected query invites
   imitation of its shape. A lesson now renders as the error alone.

After the fix, h04 passes in every mode and the ablation separates cleanly — which
is how the +4.5 result above became visible at all. The first run was measuring
the memory bug, not the repair signal.

**The transferable lesson:** a memory of past corrections is not free. Every
retrieved entry is a few-shot example, and an irrelevant few-shot example actively
degrades output. The notebooks present memory as an unambiguous improvement. The
naive implementation was a regression, and it took an ablation to see it.

### Cost, measured

| | calls/question | wall clock, 22 questions |
|---|---|---|
| `none` / `execution` | 11.5 – 11.9 | ~75 s cold |
| `critique` / `both` | 22.6 – 23.4 | 233 – 292 s cold |

Turning on the critic roughly doubles both. On this evidence it is not worth it at
3B.

### Still failing

**q09** — the grading artifact from Phase 1 (extra column, correct rows), failing
by design under D6.

**h02 under `none` and `critique`** — expected: those modes cannot act on an
execution error.

### Findings

**1. Rank feedback signals by reliability, not by sophistication.** The cheapest
signal here is also the most trustworthy, because it comes from the database
rather than from a model's opinion about a model's output.

**2. Mechanical checks beat model checks where a mechanical check exists.**
`check_against_schema()` catches hallucinated columns deterministically and
suggests the right name (`salery` → `salary`), which the database's own error does
not do. See D8.

**3. An ablation that produces a null result may be measuring the wrong thing.**
The first run said "repair mode does not matter". It was actually saying "your
memory implementation is corrupting the comparison". Worth remembering before
accepting a negative result.

**Status.** Phase 2 done. Next: Phase 3 — the four generation strategies behind
one interface, measured against 95.5%.
