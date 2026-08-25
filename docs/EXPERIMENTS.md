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
