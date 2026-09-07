"""Generation strategies — six ways to turn a question into SQL.

Five come from the source notebooks; `direct` is the control they are measured
against.

    direct         1 call.  The baseline.
    react          tool-calling agent, discovers the schema itself   (notebook 1)
    chain          intent, ground, generate, verify, repair          (workflow 1)
    parallel       one draft, specialised reviewers voting           (workflow 3)
    eval_optimize  generate, grade, regenerate, repeat               (workflow 4)
    orchestrator   staff specialists to the question, synthesise     (workflow 5)

Routing between them — workflow 2, the one the notebooks never applied to SQL —
is Phase 4 and lives in `router.py`.

Phase 9 adds two more, and they belong together:

    deep             plans, inspects the database, tests queries, then submits
    deep_seeded      the same, starting from `direct`'s draft
    self_consistency n samples of the baseline prompt, voting on the answer

The deep agent spends roughly ten calls per question, so `direct` at one call is
not the baseline it has to beat - `self_consistency` at matched cost is. Adding
an expensive strategy without also adding the cheap thing that costs the same is
how a strategy gets credit for its budget rather than its design.
"""

from __future__ import annotations

from .base import Draft, Strategy, StrategyContext
from .chain import ChainStrategy
from .deep import DeepAgentStrategy, DeepSeededStrategy
from .direct import DirectStrategy
from .eval_optimize import EvaluatorOptimizerStrategy
from .orchestrator import OrchestratorStrategy
from .parallel import ParallelStrategy
from .react import ReactStrategy
from .self_consistency import SelfConsistencyStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    "direct": DirectStrategy,
    "react": ReactStrategy,
    "chain": ChainStrategy,
    "parallel": ParallelStrategy,
    "eval_optimize": EvaluatorOptimizerStrategy,
    "orchestrator": OrchestratorStrategy,
    "deep": DeepAgentStrategy,
    "deep_seeded": DeepSeededStrategy,
    "self_consistency": SelfConsistencyStrategy,
}


def get_strategy(name: str) -> Strategy:
    """Build a strategy by name."""
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; available: {', '.join(STRATEGIES)}")
    return STRATEGIES[name]()


__all__ = [
    "Draft",
    "Strategy",
    "StrategyContext",
    "STRATEGIES",
    "get_strategy",
    "DirectStrategy",
    "ReactStrategy",
    "ChainStrategy",
    "ParallelStrategy",
    "EvaluatorOptimizerStrategy",
    "OrchestratorStrategy",
    "DeepAgentStrategy",
    "DeepSeededStrategy",
    "SelfConsistencyStrategy",
]
