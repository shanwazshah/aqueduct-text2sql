# Decision log

Every meaningful fork in the build, with the option that was rejected and why.
Written at the time the decision was made, not reconstructed afterwards.

---

### D1 — SQLite as the primary database, not MySQL

**Date:** 2026-08-25

The source notebooks run MySQL, installed via `apt-get` on a Colab Linux VM. That
does not transfer to a Windows laptop without installing MySQL Server or Docker
first, and neither earns its cost here.

SQLite also happens to be BIRD's native format, so the benchmark loads without a
conversion step.

**Rejected:** MySQL (setup burden, no benefit at this stage), PostgreSQL (adds
Docker as a hard prerequisite).

**Cost of being wrong:** low. All access goes through SQLAlchemy and the dialect
is read from the engine rather than hardcoded, so moving to MySQL is a change to
`AQ_DB_URL` plus a re-run of the test suite.

---

### D2 — Safety enforced by parsing, not by prompting

**Date:** 2026-08-25

The notebooks put `"NEVER run DELETE, DROP, UPDATE, INSERT"` in the agent persona.
That is a request. A 3B model under an awkward prompt will eventually ignore it,
and the failure mode is data loss.

`db/safety.py` parses every generated query into an AST with `sqlglot` and:
1. rejects anything that is not exactly one statement,
2. requires the root node to be a read (`SELECT` / `UNION` / `WITH…SELECT`),
3. walks the whole tree for write or admin nodes at any depth,
4. injects or tightens a `LIMIT`.

**Rejected:** a regex/keyword blocklist. It loses to `SELECT 1; DROP TABLE x`, to
comment injection, and to any write nested inside a CTE — all three of which are
in `tests/test_safety.py` and all three of which a keyword filter would miss.

**Also rejected:** relying solely on a read-only database user. That is a good
*additional* control and worth adding for a real deployment, but it gives the
Fixer agent no usable feedback — the query just fails with a permissions error
instead of an explanation of what was wrong.

---

### D3 — Comments stripped from emitted SQL, and output re-verified

**Date:** 2026-08-25

While testing D2 I noticed sqlglot rewrites `--` line comments into `/* */`
blocks. That raised the question of whether a payload containing `*/` could
terminate the comment early and smuggle in a second statement.

Tested directly: sqlglot escapes it as `* /`, so the attack fails. The behaviour
is correct.

Kept the mitigation anyway. Comments carry no value for us, so `comments=False`
on generation removes the question entirely rather than depending on a library's
escaping remaining correct across upgrades. Added a re-parse check on our own
emitted string, closing the gap between *what we validated* and *what we execute*.

`tests/test_safety.py::test_comment_payload_cannot_break_out` pins the behaviour
so a dependency bump cannot regress it silently.

---

### D4 — One OpenAI-compatible client for every backend

**Date:** 2026-08-25

Ollama (laptop) and vLLM (Kaggle T4) both expose an OpenAI-shaped `/v1` endpoint.
So the backend is a `base_url` in config, and nothing in `agent/` or
`strategies/` knows which tier it is running on.

This is what makes the two-tier plan work: develop against a 3B model locally for
fast iteration, then run the benchmark against a 7B on Kaggle without touching
agent code. Model size becomes a measurable axis rather than a rewrite.

**Rejected:** provider-specific SDKs for each backend (duplicate code paths, and
the eval harness could no longer compare like with like).

**Expected wrinkle that did not materialise.** Structured output looked like the
place the backends would diverge — Ollama's native API takes a JSON Schema in
`format`, vLLM has `guided_json`. Tested before building around it, and Ollama's
OpenAI-compatible `/v1` endpoint accepts standard
`response_format: {"type": "json_schema", ...}` directly. So no adapter layer was
needed, and `llm/structured.py` was never written — the schema goes straight
through `llm/client.py` for every backend.

Worth recording as a decision *not* taken: the abstraction was planned, tested
for, and turned out to be unnecessary. Building it anyway would have added a
layer with one implementation behind it.

---

### D5 — A demo schema with deliberate traps

**Date:** 2026-08-25

The notebooks' single flat `employees` table cannot exercise an agent crew — with
nothing to join, the Lead agent never has a reason to spin up a specialist, so
routing can never be observed doing anything.

The demo schema is five tables with joins that matter, plus four traps chosen to
be the mistakes small models actually make:

| Trap | What it catches |
|---|---|
| `orders.amount` vs `order_items.price` | picking a plausible-but-wrong column |
| `name` on three tables | unqualified references in joins |
| `employees.manager_id` self-reference | self-join reasoning |
| `employees.department_id` nullable | `JOIN` vs `LEFT JOIN` (the contractor disappears) |

These are the failures the critic and repair agents exist to catch. Without them
in the fixture, those agents would always pass and we would learn nothing.

---

### D6 — Grading compares result sets positionally, matching BIRD

**Date:** 2026-08-25

Two queries that look nothing alike can be equally correct, so grading runs both
the generated and the reference query and compares what comes back. That part was
never in question. What column order should mean took three attempts.

The intuition was that column order should not matter — `name, count` and
`count, name` answer the same question, and failing one for formatting measures
the wrong thing. Two implementations tried to encode that:

1. **Sort cells within each row.** Collapses `(min=5, max=10)` and
   `(min=10, max=5)` into the same tuple. False pass.
2. **Sort whole columns into a canonical order.** Preserves which value sits in
   which column, so it survives the multi-row case — but for a *single-row*
   result those two answers are the same multiset, and it produces the identical
   false pass. Caught by the regression test written for attempt 1.

The second failure is the informative one. The problem was not the algorithm, it
was the goal: once column names are discarded, column order carries the only
information separating those two answers, so *any* scheme that throws it away
must score them equal. Column names cannot rescue it either — a model writing
`AVG(salary)` where the reference writes `avg_sal` would then fail for naming
rather than for being wrong.

**Decision:** compare positionally, exactly as BIRD's and Spider's official
evaluation scripts do. Row order is relaxed only when neither query has an
`ORDER BY`. Numbers are compared to a tolerance of 1e-6, since two algebraically
identical `AVG` queries can differ in the last bits and failing that would be
measuring floating point rather than SQL.

**Consequence, accepted:** returning `name, salary` when the reference returns
`name` scores as wrong (this is demo question q09). Arguably the richer answer is
more useful. But relaxing it would inflate our scores relative to published
baselines and make any comparison to them meaningless — and a metric that
flatters itself is worth nothing in review.

Both false-pass cases are pinned in `tests/test_metrics.py`.

---

### D7 — The first crew is two agents, on purpose

**Date:** 2026-08-25

Phase 1 ships Writer plus Runner and nothing else — no critics, no repair, no
Lead deciding headcount. That is not an unfinished version of the real crew; it
is the control group.

The entire claim of this project is that spinning up more agents buys accuracy
worth its cost. That claim is unfalsifiable without a measured two-agent baseline
to compare against. Building the full crew first and measuring afterwards would
leave no way to attribute any of the result.

**Baseline recorded:** 91.7% EX (11/12) on the demo set, `qwen2.5-coder:3b`.
Every later phase is measured against that number.

---

### D8 — Column existence is checked in Python, not by the model

**Date:** 2026-08-25

The source notebook's critique prompt asks the model to return `schema_ok`:
does every table and column in this query exist? Phase 1 measured what that is
worth at 3B scale — shown `SELECT dept FROM employees` and told the real column
is `department`, the model returned `schema_ok: true, confidence: 0.9`.

Column existence is a set-membership test. It has a correct answer that can be
computed. Routing it through a language model converts a decidable question into
a probabilistic one and pays 18 seconds for the privilege.

`check_against_schema()` resolves aliases, extracts table and column references,
and diffs them against the introspected schema. The model is asked only what is
genuinely semantic: *does this query answer the question asked?* — wrong
aggregate, wrong join direction, missing filter. Things with no mechanical test.

**Bonus that turned out to matter:** because the check runs in Python it can
produce `column 'salery' does not exist. Did you mean 'salary'?`. The database's
own error says only `no such column: salery`. The Fixer repairs far more reliably
when told what to write instead of merely what was wrong.

**Refinement after first run:** suggestions were initially drawn from every
column in the database, which matched `dept` to `budget` — a column on a
different table. Candidates are now scoped to the table actually being queried
when the query touches only one.

---

### D9 — Error memory stores rules, not examples, and retrieval has a threshold

**Date:** 2026-08-25

Reflexion-style memory was built as the notebooks describe it: when a repair
succeeds, store the question, the broken SQL, the error, and the corrected SQL;
retrieve similar entries later and put them in the Writer's prompt.

**It made things worse, and the ablation caught it.** Question h04 (revenue by
department, a four-table join) passed with no memory and failed once a lesson
was in the store. Isolated and reproduced directly:

| condition | h04 |
|---|---|
| memory disabled | correct |
| one lesson from h02 in memory | **wrong** |

The recalled lesson was from *"which employees have never handled an order?"* —
retrieved because both questions mention orders and employees. Its corrected
query used `LEFT JOIN orders ... WHERE ... IS NULL`. The Writer reproduced that
join shape on h04 and invented a `WHERE o.status = 'shipped'` filter nobody had
asked for.

Two distinct faults, both fixed:

1. **Retrieval was too permissive.** Any shared keyword admitted a lesson. Now
   scored by Jaccard overlap with a 0.25 floor, so topical adjacency is not
   mistaken for relevance. A lesson that does not clearly apply is worse than no
   lesson — it occupies context and steers the model toward a pattern that does
   not fit.
2. **Lessons were rendered as SQL.** Showing a corrected query invites imitation
   of its shape. A lesson now renders as the error alone —
   `- no such column: oi.employee_id`. The transferable knowledge is the
   constraint, not the query that satisfied it.

Both are pinned as regression tests in `tests/test_critic.py`.

**The general lesson, worth more than the fix:** a memory of past corrections is
not free. Every retrieved entry is a few-shot example, and an irrelevant few-shot
example actively degrades output. The notebooks present memory as a
straightforward improvement; measured on a small model, the naive version was a
regression.

**Also changed:** lessons are recorded only when the repair is *verified* — the
first attempt failed and the last succeeded. The notebook records a lesson
whenever the loop iterated more than once, which files the crew's guesses next to
its knowledge.
