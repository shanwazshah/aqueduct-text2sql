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

## 2026-08-26 — Phase 3: six strategies compared, and a negative result

**What runs now.** Six generation strategies behind one `Strategy` interface,
with execution and repair held constant in the Crew so differences are
attributable to generation alone.

### Leaderboard — 22 questions, `qwen2.5-coder:3b`, execution-only repair

| strategy | EX | correct | calls/q | vs `direct` |
|---|---|---|---|---|
| **`direct`** | **95.5%** | 21/22 | **1.0** | — |
| `parallel` | 95.5% | 21/22 | 3.7 | +0.0 |
| `eval_optimize` | 95.5% | 21/22 | 5.3 | +0.0 |
| `chain` | 68.2% | 15/22 | 4.5 | **−27.3** |
| `orchestrator` | 45.5% | 10/22 | 5.8 | **−50.0** |

*(`react` excluded — see the invalidated result below.)*

**Nothing beat a single LLM call.** Two strategies matched it at three to five
times the cost. The two most elaborate ones were dramatically worse: the
orchestrator, with a planner and up to five specialist workers, got **less than
half** as many questions right as one prompt.

This contradicts the premise of the source notebooks, which present these
patterns as improvements to Text-to-SQL.

### Why decomposition loses here

Every stage that consumes another stage's output inherits its errors and cannot
detect them. The orchestrator's synthesiser is instructed to follow the workers'
findings — so when a 3B worker reports a wrong join key, the synthesiser
faithfully writes it in. Adding stages multiplies the chances of an error and
adds no mechanism for catching one.

`direct` avoids this by construction: one model, one shot, the full schema in
front of it, nothing to inherit.

**The load-bearing caveat.** This is a 3B model. These patterns come from work
with frontier models, where each stage is reliable enough that decomposition
gains from focus more than it loses from error propagation. The hypothesis this
result suggests — *decomposition helps large models and actively harms small
ones* — is exactly what the Kaggle 7B tier is for. Until that runs, the honest
claim is narrow: **at 3B, on this question set, decomposition costs accuracy.**

### The result that was wrong, and how it was caught

`react` first scored 95.5% — apparently matching `direct` at twice the cost. The
number was fabricated by a bug.

Two things in the row did not fit. `react` reported a repair on **22 of 22**
questions, where `direct` reported one. And it used 1–2 calls per question, far
too few for a loop that is supposed to list tables, inspect schemas, then query.

Running the strategy in isolation:

```
DRAFT SQL: ''
NOTES: {'steps': 0, 'executed': False}
```

Zero tool calls, zero steps, empty SQL. Every "react" answer had been written by
the Fixer from the error `Query is empty.`. **The leaderboard was scoring the
repair layer and labelling it `react`.**

Root cause, from probing the endpoint directly:

```
model: qwen2.5-coder:3b
tool_calls: None
content: {"name": "list_tables", "arguments": {}}
```

The model emits a correct tool call — as plain text in `content`. Its chat
template never tags it, so nothing populates `tool_calls`. Identical behaviour on
Ollama's native `/api/chat` and its OpenAI-compatible `/v1`, so it is the model's
template, not the protocol. `llama3.2` populates `tool_calls` properly on the
same request, and `ollama show qwen2.5-coder:3b` lists `tools` under
**Capabilities**.

**A declared capability is a claim about the model, not a guarantee about the
serving stack.**

Fixed with a fallback that recovers tool calls from message content, covering
`<tool_call>` tags, fenced JSON, arrays, and the nested `function` form, while
refusing to parse ordinary prose as a call — a final answer misread as a tool
call would make the loop never terminate. Ten regression tests.

After the fix, `react` executes real tool calls and produces real SQL. Re-running
it for a valid number.

**The general lesson:** a strategy that silently produces nothing does not look
broken when a repair layer sits behind it — it looks *fine*. Suspiciously good
numbers deserve the same scrutiny as suspiciously bad ones, and the tell here was
not the accuracy but the repair count.

### A measurement flaw to fix

The `s/q` column is not comparable across rows. `direct` had every response
cached from earlier phases and reports ~0 s; the others ran cold. Cost per
question in **calls** is sound and is what the table above reports. Wall-clock
timing needs a cache-cold run to mean anything, and that belongs with the Kaggle
sweep.

### Where strategy choice actually matters

Eight questions were solved by every strategy; **none** were solved by none. The
other fourteen were contested — which is the finding that keeps Phase 4 alive. If
no question had been contested, routing would have nothing to route.

The contested set is almost entirely the hard questions: per-group extremes,
anti-joins, multi-hop joins, self-joins, integer division. `q09` is the exception
and the most interesting row — solved *only* by `chain` and `react`, the two
strategies that produce the most minimal SQL, because the grader marks the extra
column wrong (D6).

**Status.** Phase 3 done, with the strongest result in the project so far being a
negative one. Next: Phase 4 — routing, now knowing that the thing to route
*away* from is complexity.

## 2026-08-26 — Phase 3 corrected: separating generation from rescue

The Phase 3 leaderboard above measured **strategy + repair** and reported it as
**strategy**. That is a design fault in the harness, not a detail — it lets a
strategy that generates nothing at all appear competent, because the repair layer
quietly writes the query from the error message.

Every row now records the strategy's raw draft, graded *before* repair touches
it. Same 22 questions, same models, same cache.

### Corrected leaderboard

| strategy | **gen EX** | final EX | rescued by repair | calls/q |
|---|---|---|---|---|
| **`direct`** | **90.9%** | 95.5% | 1 | **1.0** |
| `parallel` | 90.9% | 95.5% | 1 | 3.7 |
| `eval_optimize` | 90.9% | 95.5% | 1 | 5.3 |
| `chain` | 50.0% | 68.2% | 4 | 4.5 |
| `orchestrator` | 40.9% | 45.5% | 1 | 5.8 |
| `react` | **4.5%** | 81.8% | **17** | 2.5 |

**`react` generates one correct query out of twenty-two.** Its 81.8% was 17
rescues. Broken down further: it made no tool calls at all on 14 of 22 questions,
and when it *did* drive the loop it was right 4 times out of 8. The apparent
competence was entirely the Fixer.

At 3B, a multi-step ReAct loop does not work. That is notebook 1's whole
architecture, and it does not survive contact with a small model.

### The result, stated cleanly

Ranked by what the strategies actually *generate*:

```
direct / parallel / eval_optimize   90.9%     1 - 5 calls
chain                               50.0%     4.5 calls
orchestrator                        40.9%     5.8 calls
react                                4.5%     2.5 calls
```

**Generation quality falls monotonically as decomposition increases.** `parallel`
and `eval_optimize` tie with `direct` because both begin with a single-shot draft
and their extra machinery changed the answer on zero questions. Everything that
*replaces* single-shot generation with a pipeline — chain, orchestrator, react —
does worse, and the more stages it has, the worse it does.

The mechanism is the same one in every case: each stage inherits the previous
stage's errors and has no way to detect them. The orchestrator's synthesiser is
instructed to follow its workers' findings, so a wrong join key from a 3B worker
is written faithfully into the final query.

**The caveat stands and matters.** This is a 3B model. These patterns come from
work with frontier models, where each stage is reliable enough that focus gains
outweigh error propagation. The hypothesis — *decomposition helps large models and
harms small ones* — is what the Kaggle 7B tier exists to test.

### On finding this

This is the second measurement bug that flattered the project, after the grader's
false passes in Phase 1. Both were caught by something that did not fit rather
than by the score looking wrong:

  * Phase 1: a test written to prove the grader correct failed instead.
  * Phase 3: `react` reported repairs on 22 of 22 questions where `direct`
    reported 1.

Neither headline number looked suspicious. The metadata did. Worth building
harnesses that surface the metadata by default — the `rescued` column exists now
precisely so this class of fault is visible in the table rather than found by
accident.

### What this means for Phase 4

Routing was conceived as *"send hard questions to expensive strategies"*. The data
says there is nothing worth routing *to*: no strategy beats `direct` on
generation, and the expensive ones are far worse.

That does not kill the router, it redirects it. All 14 contested questions were
contested because of *repair*, not generation. So the live question becomes **how
much verification effort a question deserves**, not which pipeline to run — with
`direct` generation as the fixed baseline and the router deciding what to spend on
checking. That is a smaller claim than the one this project started with, and it
is the one the evidence supports.

## 2026-08-26 — Phase 4: routing, and the null result that ends the arc

**What the router became.** Phase 3 showed there is nothing to route *to* — no
strategy generates better than `direct`, and the elaborate ones generate far
worse. So generation was fixed at `direct` and the router was pointed at the only
decision the data left open: **how much verification a question deserves.**

Grounding, from Phase 2: execution repair is worth +4.5 points and costs a call
only when a query actually fails, so it is always on. Critic review costs a call
on every question. The router's job is therefore to spend the critic call where
it might pay — decided mechanically from the parsed SQL (joins, subqueries,
HAVING, division, self-joins) at zero cost, with an LLM router built alongside
for comparison.

### Result

| config | EX | verified | calls/q |
|---|---|---|---|
| **`trust-all`** | **95.5%** | 0/22 | **1.05** |
| `verify-all` | 95.5% | 22/22 | 2.05 |
| `router` (mechanical) | 95.5% | 15/22 | 1.73 |
| `llm-router` | 95.5% | 9/22 | 1.45 |

**All four identical. Every configuration missed exactly one question, q09, and
it is the grading edge case from D6.**

The router works — its decisions are sound, and it correctly identifies the
multi-hop joins, per-group extremes and ratio queries as risky. But the option it
gates has no value, so gating it can only add cost. `trust-all` is strictly
optimal: same accuracy, cheapest, no routing logic at all.

**The honest conclusion: the router is unnecessary, because the expensive option
it exists to ration is worthless at this scale.**

### The arc, stated in one place

Three phases, three null results, one consistent story:

  * **Phase 2** — model self-critique: +0.0 points for 2× the calls.
  * **Phase 3** — decomposition: generation quality falls monotonically as
    stages are added (90.9% → 50.0% → 40.9% → 4.5%).
  * **Phase 4** — routing between those options: +0.0 points for 1.6× the calls.

**At 3B, one good prompt plus execution feedback beats every multi-agent pattern
in both source notebooks.** What earns its keep is not agents: it is the schema
card (foreign keys and sample values), the safety guard, and the database's own
error messages. Everything expensive is the model; everything valuable is free.

The caveat that keeps this a finding rather than a verdict: **this is a 3B model
on 22 questions.** These patterns come from work with frontier models. Nothing
here is wasted if the picture inverts at 7B — the router is threshold-tunable,
every strategy is behind one interface, and the harness measures generation
separately from rescue. The infrastructure exists precisely so the Kaggle run can
answer the question rather than assume it.

### Two more bugs, both in my own instrumentation

**The schema check was regex-based and flagged correct queries.** It reported
output aliases (`SUM(amount) AS total_value`), string literals (`'cancelled'`),
type names (`REAL`) and functions (`CAST`) as nonexistent columns. Every one
fired on a query that was right, and each would have triggered a pointless repair
— and through the router, spent a critic call reviewing SQL with nothing wrong
with it. A keyword denylist cannot fix this, because identifiers only have
meaning in a grammatical position. Rewritten on the sqlglot AST; all false
positives gone, all real hallucinations still caught.

**`sqlglot.parse_one` does not raise on prose.** `parse_one("this is not sql")`
returns a column expression, so the router's `except` branch never fired and
garbage scored **0 — trusted**. The worst possible direction for that error to
run in. Fixed by checking the root node is actually a `SELECT`, caught by a test
written to assert the opposite.

That is now five instrumentation bugs across the project, against zero bugs found
in the agent logic. **The measuring apparatus has been consistently less reliable
than the thing being measured** — which is worth stating plainly, because the
instinct is always to debug the system rather than the ruler.

**Status.** Phases 0-4 complete. The remaining work is the part that can still
overturn the conclusion: BIRD on the Kaggle 7B tier.

## 2026-08-26 — Phase 6: BIRD mini-dev on 7B, and the gap collapses

**Setup.** 100 questions stratified from BIRD mini-dev (30 simple / 50 moderate /
20 challenging), 11 real databases, `qwen2.5-coder:7b` served by Ollama on a
Kaggle T4. Execution repair on, memory off, same grader as every earlier phase.

### Result

| strategy | gen EX | final EX | rescued by repair |
|---|---|---|---|
| **`direct`** | **41.0%** | **42.0%** | +1.0 |
| `chain` | 35.0% | 41.0% | +6.0 |
| `orchestrator` | 32.0% | 35.0% | +3.0 |

**41–42% on BIRD mini-dev is a credible 7B-class number** — published results
for models this size sit in the 25–45% band. That is the first external check
this project has had, and the harness passes it. A score near the demo set's
95% would have meant a bug.

### The headline: the decomposition penalty nearly vanished

| strategy | gap behind `direct` (gen), 3B / demo set | gap, 7B / BIRD |
|---|---|---|
| `chain` | **40.9 points** | **6.0 points** |
| `orchestrator` | **50.0 points** | **9.0 points** |

On final EX, `chain` at 41.0% is within a single point of `direct` at 42.0%. At
3B it trailed by 27.

This is the first evidence for the hypothesis Phase 3 raised: that decomposition
requires a capability threshold, and the workflow patterns in the source
notebooks fail on small models not because they are wrong but because each stage
needs to be reliable enough for the next one to build on.

### What cannot be claimed yet

**Two variables changed at once: model size *and* benchmark.** 3B was measured on
22 easy demo questions; 7B on 100 real BIRD questions. The gap narrowing is
consistent with scale closing it — and equally consistent with BIRD being hard
enough to compress every strategy toward the floor. A 41-point gap has less room
to exist when the leader is at 41% than when it is at 91%.

So the defensible statement today is *"the gap is 41 points in one setting and 6
in another"*, not *"scale closes the gap"*.

**The control that settles it** is running the same 100 BIRD questions on
`qwen2.5-coder:3b` — same data, same grader, same code, only the model differs.
Added as cell 9 of the Kaggle notebook. Until that lands, the causal claim stays
unmade.

### Repair, on a real benchmark

Execution repair earned +1.0 on `direct`, +6.0 on `chain`, +3.0 on
`orchestrator`. The Phase 2 finding holds at scale and on real data: repair is
worth points, it costs a call only when a query actually fails, and the more
error-prone the generator the more it recovers.

Notable that `chain` gains six times what `direct` does. A pipeline produces more
*executable but wrong* SQL, and execution feedback is precisely the signal that
catches it.

**Status.** The first result in this project that points *toward* the source
notebooks rather than away from them. Awaiting the 3B control before the
conclusion is stated as causal.

## 2026-08-27 — Phase 6 control: the earlier finding was mostly my test set

The previous entry reported the decomposition gap "collapsing" from 41 points to
6 and read it as evidence that scale closes it. That entry flagged the confound —
model size and benchmark had both changed — and said the causal claim would stay
unmade until a control ran.

It has now run. **The confound was doing about 90% of the work.**

### The control

The same 100 BIRD questions, same grader, same code, on `qwen2.5-coder:3b`.

| | 3B / demo set | 3B / BIRD | 7B / BIRD |
|---|---|---|---|
| `chain` gap behind `direct` | **40.9** | **9.0** | **6.0** |
| `orchestrator` gap | **50.0** | **13.0** | **9.0** |

Holding the model fixed at 3B and changing only the benchmark takes the `chain`
gap from 40.9 to 9.0. Changing the model then takes it from 9.0 to 6.0.

```
chain          total narrowing 34.9 pts
  benchmark change   31.9 pts   (91%)
  model size          3.0 pts   ( 9%)

orchestrator   total narrowing 41.0 pts
  benchmark change   37.0 pts   (90%)
  model size          4.0 pts   (10%)
```

### What this retracts

**Phase 3's headline was overfit to the demo set.** "Generation quality falls
monotonically as decomposition increases — 90.9% to 50.0% to 40.9%" described 22
questions on a five-table toy schema, not Text-to-SQL. On a real benchmark the
decomposition penalty is **6–13 points, not 41–50**.

The mechanism for the inflation is straightforward in hindsight: `direct` scored
90.9% on the demo set, which leaves 41 points of room *below* it for a gap to
occupy. At 29% there is far less room, and every strategy is compressed toward
the floor by the difficulty of the questions rather than separated by the merits
of its design.

A 22-question set was described at the time as "a smoke test, not a benchmark…
here to catch regressions". That caveat was correct and then quietly ignored the
moment the numbers looked interesting.

### What survives

**The direction is real, and small.** Scale narrows the gap for both strategies,
−3.0 and −4.0, consistent in sign across two independent pipelines. That is the
capability-threshold effect at its actual size.

**`direct` still wins at both model sizes** — 29.0% vs 20.0/16.0 at 3B, 41.0% vs
35.0/32.0 at 7B. Fewer moving parts still generates better SQL.

**Repair helps most where generation is weakest**, at both sizes:

| strategy | repair gain, 3B | repair gain, 7B |
|---|---|---|
| `direct` | +4.0 | +1.0 |
| `chain` | **+10.0** | **+6.0** |
| `orchestrator` | +2.0 | +3.0 |

A pipeline emits more *executable but wrong* SQL, and execution feedback is
precisely the signal that catches it. This has now held in every phase and on
both benchmarks — the most durable result in the project.

### One crossover worth watching

On **challenging** questions at 7B, `chain` beats `direct`: **30% vs 25%**. That
is the pattern behaving as its designers intended — hard problem, model capable
enough for decomposition to pay.

It is also 6 questions against 5, out of 20. **A hypothesis, not a finding.**
Testing it properly means the full 500-question mini-dev, where the challenging
stratum is 102 questions rather than 20.

### The methodological lesson

Two headline numbers in this project have now turned out to be artifacts:
`react`'s 95.5%, which was the repair layer wearing its name, and Phase 3's
41-point gap, which was the test set. Both were caught by a control rather than
by inspection, and in both cases the raw number looked entirely plausible.

Neither would have been caught by more careful reading of the code. What caught
them was measuring the same thing a second way.

**Status.** Phase 6 complete, including the retraction. The BIRD numbers are the
ones to quote; the demo-set numbers are for regression testing and nothing else.

## 2026-09-05 — Phase 7: an audit of the instrument, and a third false pass

No new measurement of the system in this entry. Phase 7 pointed the project's own
lesson back at itself — six instrumentation bugs against zero in the agent logic,
every one of them found by accident — and audited the measuring apparatus
deliberately for the first time.

It found two more bugs, one gap in what gets recorded, and one reporting error
that repeats the Phase 6 retraction in the two places a reader looks first.

### The grader could be opted out of

`execution_accuracy` decided row-order sensitivity like this:

```python
ordered = wants_order(gold_sql) and wants_order(predicted_sql)
```

D6 says row order is relaxed only when *neither* query sorts. The code relaxed it
when *either* did — so a prediction escaped the ordering check by simply not
sorting. On the demo database:

```
gold:       SELECT name, salary FROM employees ORDER BY salary DESC
prediction: SELECT name, salary FROM employees
grade:      match
```

Twelve rows, same values, different order, scored correct. That is the third
false pass in the grader, after the two column-order attempts recorded in D6, and
like both of those it ran in the flattering direction.

Order-sensitivity is now decided by the reference alone: the gold query is what
states whether sequence is part of the answer. A prediction that sorts when the
reference does not is still correct — pinned by its own test, because trading a
false pass for a false failure is not a fix.

### The sampler was skewed, and the first diagnosis of it was wrong

`stratified_sample` rounded each stratum's share independently. Independent
rounding does not sum to `n`: on mini-dev it overshoots for 62 of the 499
possible sample sizes and undershoots for another 62. The overshoot was corrected
by sorting on `question_id` and truncating.

The first reading was that the difficulty mix was wrong for 124 of those sizes.
That was measured against `round()` as the yardstick — and `round()` is the thing
that does not sum to `n`, so it was the wrong yardstick. Checked properly, the
sample size was always exactly `n` and every stratum always landed on its floor
or its ceiling. The mix was fine throughout.

The real defect is the truncation, and it is worse. BIRD question ids are
contiguous per database — 11 blocks, 30 to 66 questions each — so cutting the
id-sorted list always removed questions from whichever database sorts last. Never
a random drop.

Measured at n=23 over 400 seeds:

| | lowest 50 ids | highest 50 ids |
|---|---|---|
| old | 0.0498 | 0.0297 |
| new | 0.0464 | 0.0458 |

The last database appeared 40% less often than the first. Sample size correct,
difficulty mix correct, selection biased — which is why nothing caught it, and
why the check that finally did had to be a different one from the check that
looked obvious.

Replaced with largest-remainder apportionment, which sums to `n` by construction,
so there is nothing to patch afterwards. n=100 and n=150 select the identical
questions the old code did, verified against a captured baseline across three
seeds, so Phase 6's sample is unchanged.

### gen EX cannot be re-derived from the finished sweeps

`bird_run.Row` stored `draft_correct` but not `draft_sql`. `compare.py` has
stored the draft query since Phase 3.

That asymmetry only matters when the grader changes, which it just did. Final EX
can be re-graded from the stored `sql`. **gen EX cannot** — the draft query was
never written down, so the column every claim in this project rests on can only
be recovered by re-running the sweep on GPU.

Fixed for future sweeps. The files already on disk are permanently ungradable for
that column, and the re-grader reports them as *not re-gradable* rather than
carrying the old boolean forward as though it were evidence.

### The evidence was not in the repository

`data/bird/` and `data/comparison.json` were both gitignored, so every number in
the README traced back to a file nobody cloning the project could open. The
demo-set results (84 KB) are now committed, and the BIRD sweep outputs will be
when they are added. The questions file and the 346 MB of databases stay out;
they are third-party and downloaded.

### The retraction repeated itself in the README

The headline table is introduced as "100 questions from BIRD mini-dev". Three of
its four rows were. The `react` row was the 22-question demo set — 4.5% is 1/22 —
and `react` was never run on BIRD at all. The footnote said "22 questions"; the
table did not.

The UI was worse. Every calls-per-question figure in its sidebar was a demo-set
average, rendered beside a metric captioned "BIRD EX".

Conflating a demo-set number with a BIRD number is exactly what the Phase 6
control retracted. Nine days later it was sitting in the two places a reader looks
first, in a README whose subject is that retraction.

### What the re-grade says

Applying a grader change to finished work costs no GPU: the results files hold
the SQL, so the queries can simply be run again. `eval/regrade.py` does that.

Demo set, all six strategies, 132 rows:

| strategy | gen was | gen now | final was | final now | moved |
|---|---|---|---|---|---|
| `direct` | 90.9% | 90.9% | 95.5% | 95.5% | 0 |
| `parallel` | 90.9% | 90.9% | 95.5% | 95.5% | 0 |
| `eval_optimize` | 90.9% | 90.9% | 95.5% | 95.5% | 0 |
| `chain` | 50.0% | 50.0% | 68.2% | 68.2% | 0 |
| `orchestrator` | 40.9% | 40.9% | 45.5% | 45.5% | 0 |
| `react` | 4.5% | 4.5% | 81.8% | 81.8% | 0 |

**No grade moved.** The demo-set numbers stand exactly as published.

<!-- ─── OPEN: the BIRD re-grade ────────────────────────────────────────────
     Not run yet. bird_results_3b.json and bird_results_7b.json are Kaggle
     output and are not in the repository, and the re-grade needs the BIRD
     databases from dev.zip.

         python -m aqueduct.eval.regrade \
             --results data/bird/bird_results_7b.json \
             --databases <directory of .sqlite files>

     Fill in the final EX column when it runs. gen EX will report as not
     re-gradable: those files predate draft_sql.
─────────────────────────────────────────────────────────────────────────── -->

**BIRD, 7B and 3B: not yet re-graded.** What each outcome would mean is written
down here first, so the number cannot be read as favourably as it happens to
land:

| if | then |
|---|---|
| no grade moves | Phase 6 stands as published, and the ordering rule never bound on this benchmark |
| a few move | those numbers are corrected here and in the README, with the size of the correction stated beside them |
| many move | final EX is corrected, but **gen EX cannot be**, and the Phase 6 comparison has to be re-run before it can be quoted again |

The third outcome is the one to watch for, and it is the direct cost of not
having stored the draft query.

There is a reason to expect the first: most BIRD ranking questions carry a
`LIMIT`, so a prediction that drops `ORDER BY` usually returns a different set of
rows and failed already. That is a reason to expect a small number. It is not a
reason to skip measuring it — the 41-point gap also had a plausible story
attached.

### Two more bugs, and a change in how they were found

That is eight instrumentation bugs across the project, against zero found in the
agent logic. The grader and the sampler are the new ones. The missing `draft_sql`
is a gap in the record rather than a bug, and the README table is a reporting
error.

The pattern holds, with one change worth noting. Every earlier bug was caught by
measuring something a second way and noticing the two answers disagreed. These
two were caught by reading the code against what the documentation claimed it did
— D6 and the `stratified_sample` docstring both described behaviour the code did
not have. That is a far cheaper check than a control run, and in seven phases it
had never been done.

It is not a replacement for measuring, though, and this phase demonstrated that
on itself: the first account of the sampler bug was wrong, and what corrected it
was measuring, not re-reading.

**Also fixed, being small.** A fresh clone could not run the tests — `demo.db` is
gitignored and nothing seeded it, so `pytest` failed before an assertion ran. The
Kaggle notebook cloned an unpinned branch, so no sweep could be tied to a
revision of the code; it now checks out a named commit and prints the SHA either
way. D6 claimed a float tolerance of 1e-6 that the code does not implement — it
rounds to six decimal places, and `0.4999995` and `0.5000004` are 9e-7 apart and
compare unequal. The behaviour is kept, the description corrected.

**Status.** Phase 7 complete except the BIRD re-grade. 188 tests, up from 128.
No agent logic was touched and no conclusion has changed: the demo-set numbers
are confirmed unmoved, and the BIRD numbers are pending one offline pass that
costs no GPU.
