# Running BIRD mini-dev on Kaggle

The question this run exists to answer:

> Phases 2–4 found that on a 3B model, every multi-agent pattern from the source
> notebooks either matched or badly underperformed a single LLM call. Those
> patterns come from work with frontier models. **Does the picture invert at 7B?**

Everything else here is in service of getting a trustworthy answer to that.

---

## Before you start

**Notebook settings** (right-hand panel):

| setting | value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **On** (required — model and database downloads) |
| Persistence | Files only |

Budget: 30 GPU-hours/week, 12-hour session cap. The runner checkpoints after
every question and resumes automatically, so a lost session costs time, not work.

---

## Getting the code across

Run this **locally** first, then upload `aqueduct-src.zip` via
*Kaggle → Datasets → New Dataset*. Name it `aqueduct-src`.

```bash
cd "D:/Claude projects/Text-to-SQL" && python -m zipfile -c aqueduct-src.zip src tests pytest.ini
```

If you would rather use GitHub, push the repo and swap cell 2 for a `git clone`.

---

## Cell 1 — install

```python
!pip install -q sqlglot sqlalchemy "pydantic>=2" pydantic-settings openai 2>&1 | tail -2
print("deps ok")
```

## Cell 2 — project code

```python
import shutil, sys, zipfile, pathlib

WORK = pathlib.Path("/kaggle/working")
src_zip = next(pathlib.Path("/kaggle/input").rglob("aqueduct-src.zip"), None)
assert src_zip, "Upload aqueduct-src.zip as a Kaggle dataset first."

with zipfile.ZipFile(src_zip) as z:
    z.extractall(WORK)

sys.path.insert(0, str(WORK / "src"))
print("code at", WORK / "src")
```

## Cell 3 — serve the model

Ollama rather than vLLM: it installs in about a minute and does not fight
Kaggle's preinstalled torch. It has no continuous batching, which costs
throughput — but a lost 40-minute vLLM install costs more, and the runner is
resumable either way.

```python
import os, subprocess, time, urllib.request

!curl -fsSL https://ollama.com/install.sh | sh 2>&1 | tail -2

os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
        print("ollama up"); break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("ollama did not start")

!ollama pull qwen2.5-coder:7b 2>&1 | tail -1
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

## Cell 4 — check the model actually serves

Do not skip this. Phase 3 lost a two-hour sweep to a model that looked fine and
silently produced nothing.

```python
import json, urllib.request

def probe(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:11434/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=300).read())

r = probe({"model": "qwen2.5-coder:7b", "temperature": 0,
           "messages": [{"role": "user", "content": "Reply with the single word: ready"}]})
print("chat:", r["choices"][0]["message"]["content"][:60])

r = probe({"model": "qwen2.5-coder:7b", "temperature": 0,
           "messages": [{"role": "user", "content": "Is SELECT dept FROM employees valid if the column is department?"}],
           "response_format": {"type": "json_schema", "json_schema": {"name": "v", "schema": {
               "type": "object",
               "properties": {"ok": {"type": "boolean"}, "why": {"type": "string"}},
               "required": ["ok", "why"], "additionalProperties": False}}}})
print("structured:", r["choices"][0]["message"]["content"][:120])
```

## Cell 5 — data

`dev.zip` is 346 MB and holds all 11 mini-dev databases.

```python
import pathlib, subprocess

BIRD = pathlib.Path("/kaggle/working/data/bird")
BIRD.mkdir(parents=True, exist_ok=True)

!curl -sL "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/data/mini_dev_sqlite-00000-of-00001.json" -o {BIRD}/mini_dev_sqlite.json
!curl -sL "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" -o /kaggle/working/dev.zip
!cd /kaggle/working && unzip -q -o dev.zip -d bird_dev && rm dev.zip

# dev.zip has been repackaged more than once and sometimes nests another zip.
for inner in pathlib.Path("/kaggle/working/bird_dev").rglob("*.zip"):
    subprocess.run(["unzip", "-q", "-o", str(inner), "-d", str(inner.parent)], check=False)

dbs = sorted(pathlib.Path("/kaggle/working/bird_dev").rglob("*.sqlite"))
print(f"{len(dbs)} databases")
for d in dbs[:12]:
    print("  ", d.stem, f"{d.stat().st_size / 1e6:.0f} MB")
```

## Cell 6 — point the harness at the 7B model

```python
import os
os.environ["AQ_BASE_URL"]     = "http://127.0.0.1:11434/v1"
os.environ["AQ_API_KEY"]      = "ollama"
os.environ["AQ_MODEL_SQL"]    = "qwen2.5-coder:7b"
os.environ["AQ_MODEL_CRITIC"] = "qwen2.5-coder:7b"
os.environ["AQ_MODEL_LEAD"]   = "qwen2.5-coder:7b"
os.environ["AQ_MODEL_ANALYST"]= "qwen2.5-coder:7b"
os.environ["AQ_REQUEST_TIMEOUT"] = "600"

from aqueduct.eval.bird import load_questions, stratified_sample, describe, find_databases
import pathlib

questions = load_questions(pathlib.Path("/kaggle/working/data/bird/mini_dev_sqlite.json"))
sample    = stratified_sample(questions, 100, seed=0)
databases = find_databases(pathlib.Path("/kaggle/working/bird_dev"))

print(describe(sample))
print("databases:", len(databases))
missing = sorted({q.db_id for q in sample} - set(databases))
print("MISSING:", missing or "none")
```

## Cell 7 — the run

`direct` first. It is the control group and answers the headline question on its
own; the comparison strategies only mean something relative to it.

```python
from aqueduct.eval.bird import db_url_for, schema_for
from aqueduct.eval.bird_run import run, report
from aqueduct.crew import RepairMode
import pathlib

rows = run(sample, ["direct"], databases,
           repair=RepairMode.EXECUTION,
           path=pathlib.Path("/kaggle/working/bird_results.json"))
print(report(rows))
```

## Cell 8 — does decomposition invert at 7B?

The actual experiment. Expect roughly 2–3 hours; it resumes if the session drops.

```python
rows = run(sample, ["chain", "orchestrator"], databases,
           repair=RepairMode.EXECUTION,
           path=pathlib.Path("/kaggle/working/bird_results.json"))
print(report(rows))
```

## Cell 9 — bring the results home

```python
import json, shutil
shutil.copy("/kaggle/working/bird_results.json", "/kaggle/working/bird_results_final.json")
rows = json.load(open("/kaggle/working/bird_results_final.json"))
print(len(rows), "rows — download bird_results_final.json from the Output panel")
```

---

## What to expect

Phase 3 on the 3B model, for reference:

| strategy | gen EX | final EX |
|---|---|---|
| `direct` | 90.9% | 95.5% |
| `chain` | 50.0% | 68.2% |
| `orchestrator` | 40.9% | 45.5% |

Those are demo-set numbers on 22 easy questions. **BIRD will be far lower** —
published 7B-class results sit in the 25–45% range, and anything near 90% would
mean the harness is broken, not that the model is extraordinary.

The number that matters is not the absolute EX. It is **the gap between `direct`
and the decomposed strategies**:

- gap stays wide → decomposition hurts small models generally, and the Phase 3
  conclusion holds beyond 3B;
- gap closes or reverses → decomposition needs a capability threshold, and the
  notebooks are right about frontier models and wrong about this hardware.

Either outcome is a result. The second one is more interesting, and it is the
reason for running this at all.

## If something breaks

| symptom | cause |
|---|---|
| `MISSING: [...]` in cell 6 | `dev.zip` layout changed; check the `*.sqlite` listing from cell 5 |
| every question fails on `no such table` | databases resolved but `db_id` does not match the file stem |
| `structured:` prints prose in cell 4 | the model ignored `response_format`; strategies using structured output will fail |
| accuracy near 90% | almost certainly a harness bug, not a good model |
