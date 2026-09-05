"""Tests for the question sampler.

It decides which questions a benchmark run asks, and therefore what every
number in this project means, while never appearing in one. It had no test
before Phase 7.
"""

from __future__ import annotations

import collections
import math
import random

import pytest

from aqueduct.eval.bird import BirdQuestion, _apportion, stratified_sample

# BIRD mini-dev's own difficulty mix. Used rather than the real file because
# `data/bird/` is gitignored, and a test that only runs when a 278 KB download
# happens to be present is not a test.
MIX = {"simple": 148, "moderate": 250, "challenging": 102}
TOTAL = sum(MIX.values())


def corpus(mix: dict[str, int] = MIX, seed: int = 99) -> list[BirdQuestion]:
    """A synthetic question set with a known mix and interleaved ids.

    Difficulties are shuffled across the id range, as they are in mini-dev.
    Building them in difficulty order instead would make `question_id` a proxy
    for difficulty and quietly weaken every test below.
    """
    labels = [d for d, count in mix.items() for _ in range(count)]
    random.Random(seed).shuffle(labels)
    return [
        BirdQuestion(
            question_id=i,
            db_id=f"db{i // 50}",          # contiguous blocks, as BIRD has
            question="q",
            evidence="",
            gold_sql="SELECT 1",
            difficulty=difficulty,
        )
        for i, difficulty in enumerate(labels)
    ]


QUESTIONS = corpus()


def mix_of(sample: list[BirdQuestion]) -> dict[str, int]:
    return dict(collections.Counter(q.difficulty for q in sample))


# ── apportionment ────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 2, 3, 17, 19, 22, 23, 27, 43, 100, 150, 202, 375, 499])
def test_apportionment_sums_to_exactly_n(n):
    """The property the old implementation lacked.

    Rounding each stratum independently does not sum to `n`, which is why the
    old code needed a top-up or a truncation afterwards - and it was the
    truncation that did the damage.
    """
    assert sum(_apportion(n, MIX).values()) == n


@pytest.mark.parametrize("n", [1, 7, 19, 23, 100, 150, 202, 499])
def test_every_stratum_gets_its_floor_or_its_ceiling(n):
    quotas = _apportion(n, MIX)
    for stratum, size in MIX.items():
        exact = n * size / TOTAL
        assert math.floor(exact) <= quotas[stratum] <= math.ceil(exact)


def test_apportionment_never_exceeds_a_stratum():
    """A stratum cannot be asked for more questions than it holds."""
    tiny = {"simple": 1, "moderate": 2, "challenging": 97}
    for n in range(1, 100):
        quotas = _apportion(n, tiny)
        assert all(quotas[s] <= tiny[s] for s in tiny), (n, quotas)


# ── sampling ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 5, 17, 19, 22, 23, 27, 39, 100, 150, 202, 499])
def test_sample_is_exactly_the_requested_size(n):
    assert len(stratified_sample(QUESTIONS, n, seed=0)) == n


@pytest.mark.parametrize("n", [3, 19, 23, 100, 150, 202])
def test_sample_keeps_the_difficulty_mix(n):
    got = mix_of(stratified_sample(QUESTIONS, n, seed=0))
    for stratum, size in MIX.items():
        exact = n * size / TOTAL
        assert math.floor(exact) <= got.get(stratum, 0) <= math.ceil(exact)


def test_the_documented_sweep_size_reproduces():
    """100 questions is 30 / 50 / 20. Phase 6 says so, and the results depend on it."""
    assert mix_of(stratified_sample(QUESTIONS, 100, seed=0)) == {
        "simple": 30,
        "moderate": 50,
        "challenging": 20,
    }


def test_sampling_is_deterministic_given_a_seed():
    a = [q.question_id for q in stratified_sample(QUESTIONS, 100, seed=0)]
    b = [q.question_id for q in stratified_sample(QUESTIONS, 100, seed=0)]
    assert a == b


def test_different_seeds_choose_differently():
    a = [q.question_id for q in stratified_sample(QUESTIONS, 100, seed=0)]
    b = [q.question_id for q in stratified_sample(QUESTIONS, 100, seed=1)]
    assert a != b


def test_asking_for_everything_returns_everything():
    assert len(stratified_sample(QUESTIONS, TOTAL, seed=0)) == TOTAL
    assert len(stratified_sample(QUESTIONS, TOTAL + 50, seed=0)) == TOTAL


def test_non_positive_sample_is_empty():
    assert stratified_sample(QUESTIONS, 0, seed=0) == []
    assert stratified_sample(QUESTIONS, -5, seed=0) == []


def test_sample_is_ordered_by_question_id():
    ids = [q.question_id for q in stratified_sample(QUESTIONS, 100, seed=0)]
    assert ids == sorted(ids)


def test_high_question_ids_are_not_systematically_dropped():
    """The real Phase 7 regression.

    The old implementation over-selected for many sample sizes, then corrected
    by sorting on `question_id` and truncating - so the questions it discarded
    were always the highest ids, never a random draw. BIRD ids are contiguous
    per database, so that quietly under-sampled whichever database sorts last.

    Measured on the real mini-dev at n=23 over 400 seeds, the old code included
    the lowest 50 ids at 0.0498 and the highest 50 at 0.0297 - the last database
    appeared 40% less often than the first. The sample size and the difficulty
    mix both looked correct the whole time, which is why nothing caught it.
    """
    n, seeds, edge = 23, 120, 50
    hits: collections.Counter = collections.Counter()
    for seed in range(seeds):
        for q in stratified_sample(QUESTIONS, n, seed=seed):
            hits[q.question_id] += 1

    ids = sorted(q.question_id for q in QUESTIONS)
    low = sum(hits[i] for i in ids[:edge]) / (edge * seeds)
    high = sum(hits[i] for i in ids[-edge:]) / (edge * seeds)

    assert high / low > 0.85, f"high ids under-sampled: low={low:.4f} high={high:.4f}"
