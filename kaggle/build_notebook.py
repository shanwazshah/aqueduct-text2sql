"""Generate the Kaggle notebook.

Written as a generator rather than a hand-maintained .ipynb because the notebook
is derived from what the setup actually needs, and a JSON file full of escaped
source strings is unpleasant to review in a diff.

Every workaround here was earned by a failure on a real Kaggle session:

  * `zstd` is installed before Ollama. From v0.33 Ollama ships `.tar.zst`, and
    Kaggle's image has no zstd, so the installer aborts mid-extract and the only
    symptom is a bare `FileNotFoundError: 'ollama'` several cells later.
  * The code arrives by `git clone`, not a dataset upload. Kaggle extracts
    uploaded archives, so the `.zip` a cell looks for is never there.
  * Downloads are checked for size and exit code rather than piped to `tail`,
    which hid the original error for three rounds of debugging.
  * The model pull is its own cell so its progress is visible.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "https://github.com/shanwazshah/aqueduct-text2sql.git"
MODEL = "qwen2.5-coder:7b"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


CELLS = [
    md(f"""
# Aqueduct — BIRD mini-dev on a 7B model

**Before running:** right-hand panel → Accelerator **GPU T4 x2**, Internet **On**.

Then *Run All*. Total runtime is roughly 40 minutes to the first result
(cell 7), plus 2–3 hours for the comparison (cell 8).

## The question this answers

On a 3B model, every multi-agent pattern from the source notebooks either
matched or badly underperformed a single LLM call:

| strategy | generation EX |
|---|---|
| `direct` | 90.9% |
| `chain` | 50.0% |
| `orchestrator` | 40.9% |

Those patterns come from work with frontier models. **Does the picture invert
at 7B?** That is what cells 7 and 8 measure.

Expect BIRD scores far below the demo numbers — published 7B-class results sit
around 25–45%. Anything near 90% means the harness is broken, not that the
model is remarkable.
"""),

    md("## 1 · Dependencies and project code"),
    code(f"""
!pip install -q sqlglot sqlalchemy "pydantic>=2" pydantic-settings openai 2>&1 | tail -2

import sys, subprocess
subprocess.run(["rm", "-rf", "/kaggle/working/aq"], check=False)
subprocess.run(["git", "clone", "-q", "{REPO}", "/kaggle/working/aq"], check=True)
sys.path.insert(0, "/kaggle/working/aq/src")

import aqueduct
print("code ready:", aqueduct.__file__)
"""),

    md("""
## 2 · Ollama

`zstd` goes in first. Ollama ships its Linux release as `.tar.zst` and Kaggle's
image has no zstd, so the installer aborts during extraction — surfacing much
later as a bare `FileNotFoundError: 'ollama'`.
"""),
    code("""
import os, shutil, subprocess, time, urllib.request

subprocess.run("apt-get -qq update && apt-get -qq install -y zstd",
               shell=True, capture_output=True)
print("zstd:", shutil.which("zstd"))

r = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",
                   shell=True, capture_output=True, text=True)
print("installer exit:", r.returncode)
if r.returncode != 0:
    print(r.stdout[-800:], r.stderr[-800:])

os.environ["PATH"] = "/usr/local/bin:" + os.environ["PATH"]
os.environ["OLLAMA_HOST"] = "127.0.0.1:11434"
# Models go on /kaggle/working, which has far more room than the root filesystem.
os.environ["OLLAMA_MODELS"] = "/kaggle/working/ollama_models"
os.makedirs("/kaggle/working/ollama_models", exist_ok=True)

assert shutil.which("ollama"), "ollama not installed - check the installer output above"

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(90):
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=2)
        print("ollama serving"); break
    except Exception:
        time.sleep(2)
else:
    raise RuntimeError("ollama did not come up")
"""),

    md(f"## 3 · Pull `{MODEL}`\n\n4.7 GB. Its own cell so the progress is visible."),
    code(f"""
import subprocess, sys

p = subprocess.Popen(["ollama", "pull", "{MODEL}"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    sys.stdout.write(line)
assert p.wait() == 0, "pull failed - re-run this cell, it resumes"

print(subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout)
"""),

    md("""
## 4 · Verify the model actually works

Not optional. An earlier sweep ran for two hours against a model that looked
healthy and silently produced nothing, and the wasted time was entirely
avoidable with this check.

`structured` must print JSON. If it prints a sentence, the model is ignoring
`response_format` and cells 8's strategies will fail.
"""),
    code(f"""
import json, urllib.request, urllib.error

def probe(payload):
    req = urllib.request.Request(
        "http://127.0.0.1:11434/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={{"Content-Type": "application/json"}},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=600).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {{e.code}}: {{e.read().decode()[:200]}}") from None

r = probe({{"model": "{MODEL}", "temperature": 0,
           "messages": [{{"role": "user", "content": "Reply with the single word: ready"}}]}})
print("chat      :", r["choices"][0]["message"]["content"][:60])

r = probe({{"model": "{MODEL}", "temperature": 0,
           "messages": [{{"role": "user",
                         "content": "Is SELECT dept FROM employees valid if the column is department?"}}],
           "response_format": {{"type": "json_schema", "json_schema": {{"name": "v", "schema": {{
               "type": "object",
               "properties": {{"ok": {{"type": "boolean"}}, "why": {{"type": "string"}}}},
               "required": ["ok", "why"], "additionalProperties": False}}}}}}}})
out = r["choices"][0]["message"]["content"]
print("structured:", out[:140])
assert out.strip().startswith("{{"), "model ignored the JSON schema"
print("\\nboth checks passed")
"""),

    md("""
## 5 · BIRD data

The questions are a small JSON. The databases come from BIRD's `dev.zip`
(346 MB, expanding to roughly 1.3 GB), hosted in Beijing — so it can be slow
from Kaggle. Progress and exit codes are shown rather than swallowed.
"""),
    code("""
import pathlib, shutil, subprocess, time

subprocess.run(["rm", "-f", "/tmp/ollama.tar.zst"], check=False)   # reclaim 1.4 GB

BIRD = pathlib.Path("/kaggle/working/data/bird")
BIRD.mkdir(parents=True, exist_ok=True)

q = subprocess.run(
    ["curl", "-sL", "--fail",
     "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/"
     "data/mini_dev_sqlite-00000-of-00001.json",
     "-o", str(BIRD / "mini_dev_sqlite.json")],
    capture_output=True, text=True)
print("questions:", "ok" if q.returncode == 0 else f"FAILED {q.stderr[:200]}")

zip_path = pathlib.Path("/kaggle/working/dev.zip")
if not zip_path.exists() or zip_path.stat().st_size < 300_000_000:
    print("downloading databases (346 MB, several minutes)...")
    t0 = time.time()
    d = subprocess.run(
        ["curl", "-L", "--fail", "--max-time", "3600",
         "-w", "http=%{http_code} size=%{size_download} speed=%{speed_download}B/s\\n",
         "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip", "-o", str(zip_path)],
        capture_output=True, text=True)
    print(d.stdout.strip(), f"| {time.time() - t0:.0f}s | exit {d.returncode}")
    if d.returncode != 0:
        print("STDERR:", d.stderr[-400:])

assert zip_path.exists(), "dev.zip did not download"
print(f"dev.zip: {zip_path.stat().st_size / 1e6:.0f} MB")

print("unzip:", shutil.which("unzip"))
u = subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", "/kaggle/working/bird_dev"],
                   capture_output=True, text=True)
print("unzip exit:", u.returncode, u.stderr[-300:] if u.returncode else "")

root = pathlib.Path("/kaggle/working/bird_dev")

# dev.zip has been repackaged more than once; nested archives are common.
for _ in range(3):
    inner = [z for z in root.rglob("*.zip")]
    if not inner:
        break
    for z in inner:
        subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(z.parent)], check=False)
        z.unlink(missing_ok=True)

dbs = sorted(root.rglob("*.sqlite")) + sorted(root.rglob("*.sqlite3")) + sorted(root.rglob("*.db"))
print(f"\\ndatabases found: {len(dbs)}")
for d in dbs[:12]:
    print(f"   {d.stem:<28} {d.stat().st_size / 1e6:>7.0f} MB")

if not dbs:
    print("\\nNOTHING FOUND - top of the tree:")
    for p in sorted(root.rglob('*'))[:25]:
        print("   ", p.relative_to(root))
"""),

    md("""
## 6 · Configure and choose the questions

100 questions sampled to preserve BIRD's difficulty mix, deterministic by seed.

**`MISSING` must be `none`.** A missing database makes its questions auto-fail
and quietly drags the score down, which would look like a model result rather
than a setup problem.
"""),
    code(f"""
import os, pathlib

os.environ["AQ_BASE_URL"]        = "http://127.0.0.1:11434/v1"
os.environ["AQ_API_KEY"]         = "ollama"
os.environ["AQ_MODEL_SQL"]       = "{MODEL}"
os.environ["AQ_MODEL_CRITIC"]    = "{MODEL}"
os.environ["AQ_MODEL_LEAD"]      = "{MODEL}"
os.environ["AQ_MODEL_ANALYST"]   = "{MODEL}"
os.environ["AQ_REQUEST_TIMEOUT"] = "600"

from aqueduct.eval.bird import load_questions, stratified_sample, describe, find_databases

questions = load_questions(pathlib.Path("/kaggle/working/data/bird/mini_dev_sqlite.json"))
sample    = stratified_sample(questions, 100, seed=0)
databases = find_databases(pathlib.Path("/kaggle/working/bird_dev"))

print(describe(sample))
print("databases:", len(databases))
missing = sorted({{q.db_id for q in sample}} - set(databases))
print("MISSING:", missing or "none")
assert not missing, "some databases are missing - cell 5 did not finish"
"""),

    md("""
## 7 · The headline run — `direct`

The control group. One LLM call per question, execution repair on. Roughly 40
minutes.

Checkpointed after every question: if the session drops, re-run this cell and it
resumes where it stopped.
"""),
    code("""
import pathlib
from aqueduct.eval.bird_run import run, report
from aqueduct.crew import RepairMode

RESULTS = pathlib.Path("/kaggle/working/bird_results.json")

rows = run(sample, ["direct"], databases, repair=RepairMode.EXECUTION, path=RESULTS)
print(report(rows))
"""),

    md("""
## 8 · Does decomposition invert at 7B?

The actual experiment. `chain` and `orchestrator` were 40 and 50 points behind
`direct` at 3B. If that gap narrows or reverses here, decomposition needs a
capability threshold and the source notebooks are right about frontier models
and wrong about small ones. If it holds, the 3B finding generalises.

Two to three hours. Resumable.
"""),
    code("""
rows = run(sample, ["chain", "orchestrator"], databases,
           repair=RepairMode.EXECUTION, path=RESULTS)
print(report(rows))
"""),

    md("## 9 · Save the results\n\nDownload `bird_results.json` from the Output panel and send it back."),
    code("""
import json, shutil

shutil.copy(RESULTS, "/kaggle/working/bird_results_final.json")
rows_raw = json.load(open("/kaggle/working/bird_results_final.json"))
print(len(rows_raw), "rows saved")

for strategy in sorted({r["strategy"] for r in rows_raw}):
    subset = [r for r in rows_raw if r["strategy"] == strategy]
    correct = sum(1 for r in subset if r["correct"])
    gen = sum(1 for r in subset if r["draft_correct"])
    print(f"  {strategy:<14} gen {100 * gen / len(subset):5.1f}%   "
          f"final {100 * correct / len(subset):5.1f}%   n={len(subset)}")
"""),
]


def build() -> dict:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).parent / "aqueduct_bird_kaggle.ipynb"
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out}  ({len(CELLS)} cells)")
