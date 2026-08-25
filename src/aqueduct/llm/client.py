"""The LLM client — one class for every backend.

Ollama, vLLM, and the OpenAI API all speak the same protocol, so switching
between a 3B model on a laptop and a 7B on a Kaggle T4 is a `base_url` change.
Nothing in `agents/` imports anything backend-specific.

Two call shapes:
  * `chat()`      — free text. Used for SQL, which is not JSON.
  * `structured()`— a Pydantic model in, a validated instance out. Used for every
                    decision an agent makes: verdicts, plans, classifications.

`structured()` is the important one. The source notebooks parse model output by
stripping markdown fences and catching JSONDecodeError, which means a
malformed response silently becomes a default verdict. Here the schema is passed
to the backend, which constrains decoding so non-conforming text cannot be
generated in the first place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ..config import settings
from . import cache

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when a call fails in a way the caller cannot route around."""


@dataclass
class Usage:
    """Running totals for one run, so the eval harness can report cost."""

    calls: int = 0
    cached: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(self, model: str, usage, elapsed: float, was_cached: bool) -> None:
        self.calls += 1
        self.seconds += elapsed
        self.by_model[model] = self.by_model.get(model, 0) + 1
        if was_cached:
            self.cached += 1
            return
        if usage is not None:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def summary(self) -> str:
        hit_rate = f"{self.cached}/{self.calls}" if self.calls else "0/0"
        return (
            f"{self.calls} calls ({hit_rate} cached), "
            f"{self.total_tokens} tokens, {self.seconds:.1f}s"
        )


class LLMClient:
    """A thin wrapper over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str | None = None,
        *,
        role: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        usage: Usage | None = None,
    ):
        # A role resolves to whichever model that role is configured to use,
        # which is what lets the Lead agent run on a small model while SQL
        # generation gets the biggest one the tier can afford.
        self.model = model or settings.model_for(role or "sql")
        self.temperature = settings.temperature if temperature is None else temperature
        self.usage = usage or Usage()
        self._client = OpenAI(
            base_url=base_url or settings.base_url,
            api_key=api_key or settings.api_key,
            timeout=settings.request_timeout,
        )

    # ── free text ────────────────────────────────────────────────────

    def chat(self, system: str, user: str) -> str:
        """Send a prompt, get text back."""
        key = cache.make_key(self.model, system, user, self.temperature)
        if self.temperature == 0 and (hit := cache.get(key)) is not None:
            self.usage.record(self.model, None, 0.0, was_cached=True)
            return hit

        start = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            raise LLMError(f"{self.model} call failed: {e}") from e

        elapsed = perf_counter() - start
        text = (response.choices[0].message.content or "").strip()
        self.usage.record(self.model, response.usage, elapsed, was_cached=False)

        if self.temperature == 0:
            cache.put(key, text, {"model": self.model, "seconds": round(elapsed, 2)})
        return text

    # ── structured ───────────────────────────────────────────────────

    def structured(self, system: str, user: str, schema: type[T]) -> T:
        """Send a prompt, get a validated Pydantic instance back.

        The schema is enforced by the backend during decoding, so this does not
        need the fence-stripping and JSONDecodeError handling the notebooks
        relied on. Validation is still checked — constrained decoding guarantees
        well-formed JSON matching the schema, not that the *values* are sane.
        """
        key = cache.make_key(
            self.model, system, user, self.temperature, schema.__name__
        )
        if self.temperature == 0 and (hit := cache.get(key)) is not None:
            try:
                self.usage.record(self.model, None, 0.0, was_cached=True)
                return schema.model_validate_json(hit)
            except ValidationError:
                pass  # schema changed since caching — fall through and re-ask

        start = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": _json_schema_for(schema),
                    },
                },
            )
        except Exception as e:
            raise LLMError(f"{self.model} structured call failed: {e}") from e

        elapsed = perf_counter() - start
        raw = (response.choices[0].message.content or "").strip()
        self.usage.record(self.model, response.usage, elapsed, was_cached=False)

        try:
            parsed = schema.model_validate_json(raw)
        except ValidationError as e:
            raise LLMError(
                f"{self.model} returned JSON that does not satisfy "
                f"{schema.__name__}: {e}"
            ) from e

        if self.temperature == 0:
            cache.put(key, raw, {"model": self.model, "seconds": round(elapsed, 2)})
        return parsed


def _json_schema_for(model: type[BaseModel]) -> dict:
    """Convert a Pydantic model into a schema the backend will accept.

    Pydantic emits `$defs` with `$ref` pointers for nested models. Some
    constrained-decoding backends handle refs poorly, so they are inlined here
    and `additionalProperties: false` is set throughout — which keeps the
    grammar tight and stops models padding the object with invented keys.
    """
    schema = model.model_json_schema()
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                return resolve(defs.get(name, {}))
            resolved = {k: resolve(v) for k, v in node.items()}
            if resolved.get("type") == "object":
                resolved.setdefault("additionalProperties", False)
            return resolved
        if isinstance(node, list):
            return [resolve(item) for item in node]
        return node

    return resolve(schema)


def load_json_loosely(text: str) -> dict | None:
    """Last-resort parse for text that should have been JSON but is not.

    Only used on backends that ignore `response_format`. Kept deliberately small
    — if this is being hit often, the backend is misconfigured and that is worth
    finding out rather than papering over.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
