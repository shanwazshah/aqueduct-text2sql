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

**Known wrinkle:** structured output is the one place the backends genuinely
differ — Ollama takes a JSON Schema in `format`, vLLM uses `guided_json`. That
difference is isolated to `llm/structured.py`.

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
