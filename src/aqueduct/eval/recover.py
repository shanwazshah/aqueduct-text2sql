"""Recover sweep rows from an executed notebook's output.

Phase 9's arms each ran in a fresh Kaggle session with an empty
`/kaggle/working`, so each one's results file replaced the last. Only the final
arm survived on disk and the report showed one row.

The data was not actually lost. `run()` returns its row dict, and Jupyter records
the repr of a cell's last expression — so all 150 rows were sitting in the saved
notebook, in `outputs`. This parses them back out.

That is a recovery tool, not a substitute for the results file. The repr is
truncated by Jupyter for long strings, so `sql`, `draft_sql` and `reason` come
back unreliable; the graded booleans, counts and timings do not. Everything the
headline table needs survives, and everything needed to *re-grade* does not — so
a file written by this is explicitly marked partial.

    python -m aqueduct.eval.recover notebook.ipynb -o recovered.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Jupyter writes the repr of the cell's final expression. `run()` returns
# dict[(strategy, question_id) -> Row], so each Row's fields appear as
# `field=value` pairs inside `Row(...)`.
_ROW = re.compile(r"Row\((.*?)\)(?=[,\}])", re.DOTALL)


def _field(body: str, pattern: str, cast=str, default=None):
    match = re.search(pattern, body)
    return cast(match.group(1)) if match else default


def parse_rows(notebook: Path) -> list[dict]:
    """Every row recoverable from the notebook's cell outputs."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    recovered: dict[tuple[str, int], dict] = {}

    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            if output.get("output_type") != "execute_result":
                continue
            text = "".join(output.get("data", {}).get("text/plain", []))

            for body in _ROW.findall(text):
                strategy = _field(body, r"strategy='([^']*)'")
                question_id = _field(body, r"question_id=(\d+)", int)
                if not strategy or question_id is None:
                    continue

                recovered[(strategy, question_id)] = {
                    "strategy": strategy,
                    "question_id": question_id,
                    "db_id": _field(body, r"db_id='([^']*)'", str, ""),
                    "difficulty": _field(body, r"difficulty='([^']*)'", str, ""),
                    # `(?<![a-z_])` so this does not match `draft_correct=`.
                    "correct": _field(body, r"(?<![a-z_])correct=(True|False)") == "True",
                    "draft_correct": _field(body, r"draft_correct=(True|False)") == "True",
                    "calls": _field(body, r"calls=(\d+)", int, 0),
                    "seconds": _field(body, r"seconds=([\d.]+)", float, 0.0),
                    "repaired": _field(body, r"repaired=(True|False)") == "True",
                    "agents": _agents(body),
                    # Not recoverable: the repr truncates long strings, so these
                    # would be silently partial. Empty is honest; a half-query
                    # would look re-gradable and is not.
                    "sql": "",
                    "draft_sql": "",
                    "reason": "recovered from notebook output",
                }

    return list(recovered.values())


def _agents(body: str) -> list[str]:
    raw = _field(body, r"agents=\[([^\]]*)\]", str, "")
    return [a.strip().strip("'\"") for a in raw.split(",") if a.strip()]


def report(rows: list[dict]) -> str:
    lines = [
        f"{len(rows)} rows recovered",
        "",
        f"{'arm':<20}{'gen EX':>9}{'final EX':>10}{'calls/q':>9}{'s/q':>9}"
        f"{'submitted':>12}{'n':>5}",
        "-" * 74,
    ]
    gen: dict[str, float] = {}
    for arm in sorted({r["strategy"] for r in rows}):
        subset = [r for r in rows if r["strategy"] == arm]
        n = len(subset)
        gen[arm] = 100 * sum(r["draft_correct"] for r in subset) / n
        lines.append(
            f"{arm:<20}{gen[arm]:>8.1f}%"
            f"{100 * sum(r['correct'] for r in subset) / n:>9.1f}%"
            f"{sum(r['calls'] for r in subset) / n:>9.1f}"
            f"{sum(r['seconds'] for r in subset) / n:>9.1f}"
            f"{sum(1 for r in subset if 'submit' in r['agents']):>9}/{n:<3}{n:>5}"
        )
    lines.append("-" * 74)

    if "direct" in gen:
        lines.append("")
        for arm, score in sorted(gen.items()):
            if arm != "direct":
                lines.append(f"  {arm:<20}{score - gen['direct']:>+7.1f} vs direct")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()

    rows = parse_rows(args.notebook)
    print(report(rows))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
