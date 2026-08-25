"""Disk cache for LLM responses.

The point of this is the evaluation sweep. A 200-question run across six
strategies is thousands of calls; without caching, fixing one bug in the report
generator means paying for all of them again. With it, a re-run is close to free
and only genuinely new prompts hit the GPU.

Keyed on everything that can change an answer — model, prompt, temperature,
schema — so a cache hit is a response the model would have produced anyway.
Temperature > 0 is not cached, since the variation is the point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import CACHE_DIR, settings


def make_key(
    model: str,
    system: str,
    user: str,
    temperature: float,
    schema_name: str | None = None,
) -> str:
    """Stable hash of everything that determines a response."""
    payload = json.dumps(
        {
            "model": model,
            "system": system,
            "user": user,
            "temperature": temperature,
            "schema": schema_name,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _path_for(key: str) -> Path:
    # Shard by the first two characters; a flat directory with tens of
    # thousands of files gets unpleasant to work with on Windows.
    shard = CACHE_DIR / key[:2]
    shard.mkdir(parents=True, exist_ok=True)
    return shard / f"{key}.json"


def get(key: str) -> str | None:
    """Return a cached response, or None."""
    if not settings.cache_enabled:
        return None

    path = _path_for(key)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))["response"]
    except (json.JSONDecodeError, KeyError, OSError):
        # A corrupt entry is not worth crashing over; treat it as a miss.
        return None


def put(key: str, response: str, meta: dict | None = None) -> None:
    """Store a response. Failures are ignored — a cache is an optimisation."""
    if not settings.cache_enabled:
        return

    try:
        _path_for(key).write_text(
            json.dumps({"response": response, "meta": meta or {}}, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def stats() -> dict[str, int]:
    """Entry count and total size on disk, for the eval report."""
    files = list(CACHE_DIR.rglob("*.json"))
    return {
        "entries": len(files),
        "bytes": sum(f.stat().st_size for f in files),
    }


def clear() -> int:
    """Delete every cached response. Returns how many were removed."""
    removed = 0
    for f in CACHE_DIR.rglob("*.json"):
        try:
            f.unlink()
            removed += 1
        except OSError:
            pass
    return removed
