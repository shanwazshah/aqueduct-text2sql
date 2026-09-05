"""BIRD mini-dev — the real benchmark.

500 questions across 11 databases, with human-written reference SQL and a
published difficulty label. This is what makes the project's numbers comparable
to anyone else's; the 22-question demo set exists to catch regressions quickly,
not to be believed.

Two things about BIRD that are easy to get wrong and would silently invalidate
every number:

**The `evidence` field is part of the question.** Each item carries a hint
supplying external knowledge the schema does not contain — that "eligible free
rate" means `Free Meal Count / Enrollment`, for instance. BIRD's official
evaluation feeds it to the model, and the reference SQL assumes it. Withholding
it does not make the benchmark harder in an interesting way; it makes the
questions unanswerable and the comparison meaningless.

**Every question has its own database.** There is no single connection. The
harness resolves `db_id` to a SQLite file per question, and schemas are cached
because introspecting an 11-database corpus repeatedly is slow.

Databases come from BIRD's `dev.zip` (~346 MB), which is downloaded on the
Kaggle side — see `kaggle/README.md`.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..config import DATA_DIR
from ..db.introspect import Schema, load_schema

MINI_DEV_URL = (
    "https://huggingface.co/datasets/birdsql/bird_mini_dev/resolve/main/"
    "data/mini_dev_sqlite-00000-of-00001.json"
)
DEV_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"

BIRD_DIR = DATA_DIR / "bird"

# BIRD's own labels. `simple` here is not the demo set's `easy`.
DIFFICULTIES = ("simple", "moderate", "challenging")


@dataclass(frozen=True)
class BirdQuestion:
    """One BIRD item."""

    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str

    @property
    def id(self) -> str:
        return f"bird{self.question_id}"

    @property
    def trap(self) -> str | None:
        """Kept for interface compatibility with the demo set's Question."""
        return self.db_id

    def prompt(self) -> str:
        """Question text as the model should see it.

        The evidence is appended rather than merged, so it reads as a supplied
        hint rather than as part of what is being asked.
        """
        if not self.evidence or not self.evidence.strip():
            return self.question
        return f"{self.question}\n\nHint: {self.evidence.strip()}"


def load_questions(path: Path | None = None) -> list[BirdQuestion]:
    """Read the mini-dev JSON."""
    source = path or (BIRD_DIR / "mini_dev_sqlite.json")
    if not source.exists():
        raise FileNotFoundError(
            f"BIRD questions not found at {source}. Download with:\n"
            f"  curl -L '{MINI_DEV_URL}' -o {source}"
        )

    raw = json.loads(source.read_text(encoding="utf-8"))
    return [
        BirdQuestion(
            question_id=int(item["question_id"]),
            db_id=item["db_id"],
            question=item["question"],
            evidence=item.get("evidence", "") or "",
            gold_sql=item["SQL"],
            difficulty=item.get("difficulty", "unknown"),
        )
        for item in raw
    ]


def find_databases(root: Path) -> dict[str, Path]:
    """Map `db_id` to its SQLite file.

    `dev.zip` has been repackaged more than once — sometimes
    `dev_databases/<id>/<id>.sqlite`, sometimes nested a level deeper, sometimes
    with an inner zip. Rather than hardcode a layout that breaks on the next
    release, this globs for `*.sqlite` and keys on the file's own stem.
    """
    databases: dict[str, Path] = {}
    for sqlite_file in root.rglob("*.sqlite"):
        databases.setdefault(sqlite_file.stem, sqlite_file)
    return databases


def db_url_for(path: Path) -> str:
    """SQLAlchemy URL for a database file, opened read-only.

    BIRD databases are reference data. The safety guard already blocks writes at
    the SQL level; opening read-only means even a bug in the harness cannot
    corrupt a benchmark database mid-sweep and quietly change later results.
    """
    return f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"


@lru_cache(maxsize=16)
def schema_for(db_url: str) -> Schema:
    """Introspect a database once and reuse it.

    Sample values require a query per text column, so re-introspecting a
    50-table database for every one of its questions dominates the sweep.
    """
    return load_schema(db_url)


def stratified_sample(
    questions: list[BirdQuestion],
    n: int,
    *,
    seed: int = 0,
) -> list[BirdQuestion]:
    """Take `n` questions preserving BIRD's difficulty mix.

    A random sample would drift the difficulty balance run to run, and since
    accuracy varies sharply with difficulty, that variation would be
    indistinguishable from a real effect. Sampling within each stratum keeps the
    mix fixed so two runs are comparable.

    Deterministic given `seed`, so a resumed or repeated sweep asks the same
    questions.
    """
    if n >= len(questions):
        return list(questions)
    if n <= 0:
        return []

    rng = random.Random(seed)
    by_difficulty: dict[str, list[BirdQuestion]] = {}
    for question in questions:
        by_difficulty.setdefault(question.difficulty, []).append(question)

    quotas = _apportion(n, {d: len(g) for d, g in by_difficulty.items()})

    chosen: list[BirdQuestion] = []
    for difficulty, group in sorted(by_difficulty.items()):
        chosen.extend(rng.sample(group, quotas[difficulty]))

    chosen.sort(key=lambda q: q.question_id)
    return chosen


def _apportion(n: int, sizes: dict[str, int]) -> dict[str, int]:
    """Split `n` seats across strata proportionally, summing to exactly `n`.

    Rounding each stratum independently does not sum to `n`. On mini-dev it
    overshoots for 62 of the 499 possible sample sizes and undershoots for
    another 62, and the old code patched the difference afterwards: topping up
    from the whole pool, which ignores strata, or truncating the id-sorted list,
    which deterministically dropped questions from whichever database sorts last
    (BIRD ids are contiguous per database, so that is never a random drop).
    Either way the sample no longer had the mix this function promises.

    Largest remainder gives every stratum its floor and then hands the leftover
    seats to the largest fractional parts. It sums to `n` by construction, so
    there is nothing to patch afterwards.

    For the sizes this project actually sweeps - 100 and 150 - it selects the
    identical questions the previous implementation did, so results already
    published stay reproducible.
    """
    total = sum(sizes.values())
    exact = {stratum: n * size / total for stratum, size in sizes.items()}
    quotas = {stratum: int(value) for stratum, value in exact.items()}

    # Ties broken by stratum name, so the apportionment is deterministic.
    order = sorted(sizes, key=lambda d: (-(exact[d] - int(exact[d])), d))
    leftover = n - sum(quotas.values())
    for stratum in order:
        if leftover <= 0:
            break
        if quotas[stratum] < sizes[stratum]:
            quotas[stratum] += 1
            leftover -= 1

    return quotas


def describe(questions: list[BirdQuestion]) -> str:
    """One-line summary of a question set."""
    difficulty = Counter(q.difficulty for q in questions)
    databases = Counter(q.db_id for q in questions)
    parts = [f"{len(questions)} questions", f"{len(databases)} databases"]
    parts += [f"{d}={difficulty.get(d, 0)}" for d in DIFFICULTIES]
    return "  ".join(parts)
