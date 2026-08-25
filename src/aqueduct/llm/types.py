"""Shared field types for structured LLM output.

**Constrained decoding guarantees structure, not semantics.** A JSON schema sent
as `response_format` makes the backend emit an object with the right keys and the
right types. It does not enforce `minimum` and `maximum` — those are validation
keywords, not grammar productions, and the sampler ignores them.

This surfaced the first time a chain strategy ran: a field documented as
`confidence, 0.0 to 1.0` came back as `100`. Valid JSON, correct type, correct
key, semantically nonsense — the model was thinking in percent. Pydantic then
rejected it and killed the whole question.

The failure mode is worth naming because it is easy to assume otherwise: the
schema constrains what *shape* is generated, and every remaining assumption about
*values* still needs enforcing on our side.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator


def _to_unit_interval(value):
    """Coerce a confidence-like number into 0.0-1.0.

    Models express confidence on whichever scale the prompt suggests to them, and
    sometimes on one it does not. 0-100 is the common alternative, so it is
    rescaled rather than rejected; anything still outside the range is clamped.

    Clamping beats raising here: the caller wants a number to compare against a
    threshold, and losing an entire question because a decorative field arrived
    on the wrong scale is a bad trade.
    """
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)):
        return value  # let Pydantic report the type error
    number = float(value)
    if 1.0 < number <= 100.0:
        number /= 100.0
    return min(1.0, max(0.0, number))


UnitFloat = Annotated[float, BeforeValidator(_to_unit_interval)]
"""A 0.0-1.0 score that tolerates a model using a different scale."""
