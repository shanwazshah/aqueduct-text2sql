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
# Aqueduct — does decomposition need a capable model?

**Before running, set all three in the right-hand panel:**

| setting | value |
|---|---|
| Accelerator | **GPU T4 x2** |
| Internet | **On** |
| Persistence | **Files only** |

Persistence matters. Without it a session reset deletes `/kaggle/working`, and
several hours of results go with it.

Then *Run All*. Roughly five hours end to end — about three for the 7B sweep and
two for the 3B control. Every sweep checkpoints after each question, so a dropped
session resumes rather than restarts.

## The question

On a 3B model over an easy 22-question demo set, every multi-agent pattern from
the source notebooks badly underperformed a single LLM call — `chain` by 41
points, `orchestrator` by 50.

Those patterns come from work with frontier models, which suggests they may need
a capability threshold rather than being wrong outright. **Does the gap close as
the model gets bigger?**

This notebook runs **both** model sizes over the **same 100 BIRD questions**, so
the only thing that differs is the model. An earlier attempt compared 3B on the
demo set against 7B on BIRD, and changing two variables at once left the result
unattributable.

Expect absolute scores far below the demo numbers — published 7B-class BIRD
results sit around 25–45%. Anything near 90% means the harness is broken, not
that the model is remarkable.
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
## 7 · The experiment harness

Both model sizes run the **same 100 questions**, through the same grader and the
same code. Only the model changes — which is the whole point: an earlier attempt
compared 3B on an easy demo set against 7B on BIRD, and changing two variables at
once made the result unattributable.

Each model writes to its own file and checkpoints after every question. If the
session drops, re-run the cell and it resumes.
"""),
    code("""
import importlib, os, pathlib, subprocess, sys

from aqueduct.eval.bird_run import run, report
from aqueduct.crew import RepairMode

STRATEGIES = ["direct", "chain", "orchestrator"]


def run_for_model(model: str, out_path: str):
    \"\"\"Point the whole stack at `model` and sweep the sample.\"\"\"
    p = subprocess.Popen(["ollama", "pull", model],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line)
    assert p.wait() == 0, f"pull of {model} failed - re-run, it resumes"

    for role in ("SQL", "CRITIC", "LEAD", "ANALYST"):
        os.environ[f"AQ_MODEL_{role}"] = model

    # Settings are read at import time, so the modules holding them must be
    # reloaded for new model names to take effect. The response cache keys on
    # model name, so the two models cannot contaminate each other.
    import aqueduct.config
    importlib.reload(aqueduct.config)
    import aqueduct.llm.client
    importlib.reload(aqueduct.llm.client)
    print("model now:", aqueduct.config.settings.model_sql, flush=True)

    rows = run(sample, STRATEGIES, databases,
               repair=RepairMode.EXECUTION, path=pathlib.Path(out_path))
    print(report(rows))
    return rows


print("harness ready")
"""),

    md("""
## 8 · Run the 7B model

Roughly three hours for all three strategies. Resumable.
"""),
    code("""
rows_7b = run_for_model("qwen2.5-coder:7b", "/kaggle/working/bird_results_7b.json")
"""),

    md("""
## 9 · Run the 3B model — the control

The same 100 questions on the smaller model. This is what makes the comparison
causal: if the gap between `direct` and the decomposed strategies is wide here
and narrow at 7B, scale is what closes it. If both are narrow, the benchmark was
doing the work and the earlier 3B finding was overfit to an easy demo set.

Roughly two hours.
"""),
    code("""
rows_3b = run_for_model("qwen2.5-coder:3b", "/kaggle/working/bird_results_3b.json")
"""),

    md("""
## 10 · The comparison

The number that matters is the **gap** between `direct` and the decomposed
strategies, and how that gap changes with model size.

Download both JSON files from the Output panel — they carry the per-question
detail behind this summary.
"""),
    code("""
import json, pathlib

summary = {}
for label, path in (("3B", "/kaggle/working/bird_results_3b.json"),
                    ("7B", "/kaggle/working/bird_results_7b.json")):
    f = pathlib.Path(path)
    if not f.exists():
        print(f"{label}: not run")
        continue
    raw = json.load(open(f))
    summary[label] = {}
    print(f"\\n=== {label} - {len(raw)} rows ===")
    for strategy in sorted({r["strategy"] for r in raw}):
        subset = [r for r in raw if r["strategy"] == strategy]
        gen = 100 * sum(1 for r in subset if r["draft_correct"]) / len(subset)
        fin = 100 * sum(1 for r in subset if r["correct"]) / len(subset)
        summary[label][strategy] = gen
        cuts = {}
        for level in ("simple", "moderate", "challenging"):
            g = [r for r in subset if r["difficulty"] == level]
            cuts[level] = f"{100 * sum(1 for r in g if r['correct']) / len(g):.0f}%" if g else "-"
        print(f"  {strategy:<14} gen {gen:5.1f}%  final {fin:5.1f}%  n={len(subset):<5}"
              f"simple {cuts['simple']:>5}  moderate {cuts['moderate']:>5}"
              f"  challenging {cuts['challenging']:>5}")

if len(summary) == 2:
    print("\\n=== gap behind `direct`, by model size (generation EX) ===")
    print(f"{'strategy':<16}{'3B':>8}{'7B':>8}{'change':>10}")
    for strategy in ("chain", "orchestrator"):
        g3 = summary["3B"].get("direct", 0) - summary["3B"].get(strategy, 0)
        g7 = summary["7B"].get("direct", 0) - summary["7B"].get(strategy, 0)
        print(f"{strategy:<16}{g3:>7.1f}{g7:>8.1f}{g7 - g3:>+10.1f}")
    print("\\nA gap that shrinks with scale supports the capability-threshold")
    print("reading. A gap that is already small at 3B means the earlier demo-set")
    print("result was overfit to easy questions.")
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
