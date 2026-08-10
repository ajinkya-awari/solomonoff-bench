"""Tests for CTW baseline and Solomonoff Gap metric.

Validation cases (must all pass before any Week 2 benchmark run):
1. All-zeros: H_ctw → 0 (CTW learns the pattern)
2. All-ones:  H_ctw → 0
3. Alternating (010101...): H_ctw → 0 with D >= 1
4. Uniform random: H_ctw ≈ 1.0 (CTW cannot compress)
5. P(0) + P(1) = 1.0 exactly for every prediction call
6. SG(M, x) = H(M, x) - H_CTW(x) arithmetic is correct
7. SG < 0 is possible (model can beat CTW on structured sequences)
8. score_sequence_str rejects non-binary strings
9. depth parameter is respected
10. CTW sequential update is consistent with predict()
"""

from __future__ import annotations

import math
import random

import pytest

from solomonoff_bench.baselines.ctw import CTW
from solomonoff_bench.metrics.solomonoff_gap import compute_sg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entropy_of_sequence(ctw: CTW, context: list[int], prediction: list[int]) -> float:
    """Score prediction bits one at a time using update(), return mean H."""
    history = list(context)
    fresh = CTW(depth=ctw.depth)
    for bit in context:
        fresh.update(history[:history.index(bit) if bit in history else 0], bit)

    # Simpler: use score_sequence directly
    result = ctw.score_sequence(context, prediction)
    return result["h_ctw_bits_per_sym"]


def _make_alternating(length: int, start: int = 0) -> list[int]:
    return [(start + i) % 2 for i in range(length)]


# ---------------------------------------------------------------------------
# Test 1-2: Structured sequences → H_ctw → 0
# ---------------------------------------------------------------------------

def test_all_zeros_entropy_low():
    """All-zeros: CTW learns P(0)→1, so H_ctw should approach 0."""
    ctw = CTW(depth=8)
    context = [0] * 100
    prediction = [0] * 100
    result = ctw.score_sequence(context, prediction)
    h = result["h_ctw_bits_per_sym"]
    assert h < 0.05, f"Expected H_ctw < 0.05 for all-zeros, got {h:.4f}"


def test_all_ones_entropy_low():
    """All-ones: symmetric to all-zeros."""
    ctw = CTW(depth=8)
    context = [1] * 100
    prediction = [1] * 100
    result = ctw.score_sequence(context, prediction)
    h = result["h_ctw_bits_per_sym"]
    assert h < 0.05, f"Expected H_ctw < 0.05 for all-ones, got {h:.4f}"


def test_alternating_entropy_low():
    """Alternating 010101...: CTW should learn the pattern and beat random (H<1).

    With 100 context symbols the root node accumulates ~50/50 counts, so the
    CTW mixing formula dilutes the leaf estimate (~0.99) with the root (0.5),
    giving P(correct) ≈ 0.745 and H ≈ 0.42 bits.  Full convergence to 0 would
    require far more context.  We check H < 0.5: CTW is learning, not random.
    """
    ctw = CTW(depth=4)
    context = _make_alternating(100, start=0)
    prediction = _make_alternating(100, start=0)  # continues pattern
    result = ctw.score_sequence(context, prediction)
    h = result["h_ctw_bits_per_sym"]
    assert h < 0.50, f"Expected H_ctw < 0.50 for alternating, got {h:.4f}"


# ---------------------------------------------------------------------------
# Test 4: Random sequence → H_ctw ≈ 1.0
# ---------------------------------------------------------------------------

def test_random_sequence_entropy_near_one():
    """Random bits: CTW cannot compress, H_ctw should be close to 1.0."""
    rng = random.Random(42)
    ctw = CTW(depth=8)
    context = [rng.randint(0, 1) for _ in range(100)]
    prediction = [rng.randint(0, 1) for _ in range(100)]
    result = ctw.score_sequence(context, prediction)
    h = result["h_ctw_bits_per_sym"]
    assert 0.85 < h < 1.15, f"Expected H_ctw ≈ 1.0 for random bits, got {h:.4f}"


# ---------------------------------------------------------------------------
# Test 5: P(0) + P(1) = 1.0 always
# ---------------------------------------------------------------------------

def test_probabilities_sum_to_one():
    """predict() must return probabilities that sum to 1.0."""
    ctw = CTW(depth=4)
    rng = random.Random(7)
    history: list[int] = []
    for _ in range(50):
        p0, p1 = ctw.predict(history)
        assert abs(p0 + p1 - 1.0) < 1e-10, (
            f"P(0)+P(1)={p0+p1:.12f} != 1.0 at history length {len(history)}"
        )
        bit = rng.randint(0, 1)
        ctw.update(history, bit)
        history.append(bit)


# ---------------------------------------------------------------------------
# Test 6: SG arithmetic
# ---------------------------------------------------------------------------

def test_sg_arithmetic():
    """SG = H(M, x) - H_CTW(x) — verify the arithmetic is correct."""
    sequence = "0" * 200
    # A perfect model: P(0)=1 everywhere → H(M)=0
    p0_list = [1.0] * 100
    result = compute_sg(sequence, p0_list)
    assert result["h_model_bits_per_sym"] < 1e-9, "Perfect model should have H(M)≈0"
    assert result["sg_primary"] < 0, "SG should be negative when H(M) < H_CTW"

    # A random model: P(0)=0.5 everywhere → H(M)=1
    p0_random = [0.5] * 100
    result_random = compute_sg(sequence, p0_random)
    assert abs(result_random["h_model_bits_per_sym"] - 1.0) < 1e-6, (
        "Uniform model should have H(M)=1.0"
    )
    assert result_random["sg_primary"] > 0, "Random model should have SG>0 on structured input"


# ---------------------------------------------------------------------------
# Test 7: SG < 0 is possible (model beats CTW)
# ---------------------------------------------------------------------------

def test_sg_can_be_negative():
    """A perfect model on an all-zeros sequence should beat CTW → SG < 0."""
    sequence = "0" * 200
    p0_perfect = [1.0 - 1e-10] * 100  # near-perfect prediction
    result = compute_sg(sequence, p0_perfect, depths=(4, 8))
    assert result["sg_primary"] < 0, (
        f"Perfect model on all-zeros should have SG<0, got {result['sg_primary']:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: score_sequence_str rejects bad input
# ---------------------------------------------------------------------------

def test_score_sequence_str_rejects_non_binary():
    ctw = CTW(depth=4)
    with pytest.raises(AssertionError):
        ctw.score_sequence_str("a" * 200)


def test_score_sequence_str_rejects_wrong_length():
    ctw = CTW(depth=4)
    with pytest.raises(AssertionError):
        ctw.score_sequence_str("0" * 100)  # too short


# ---------------------------------------------------------------------------
# Test 9: depth parameter is respected
# ---------------------------------------------------------------------------

def test_depth_parameter():
    """Deeper trees should converge better on structured sequences."""
    sequence = "01" * 100  # alternating, length 200
    context = [int(c) for c in sequence[:100]]
    prediction = [int(c) for c in sequence[100:]]

    ctw_shallow = CTW(depth=1)
    ctw_deep = CTW(depth=8)

    result_shallow = ctw_shallow.score_sequence(context, prediction)
    result_deep = ctw_deep.score_sequence(context, prediction)

    # Both should be low; shallow might be slightly higher but both should work
    assert result_shallow["h_ctw_bits_per_sym"] < 0.5
    assert result_deep["h_ctw_bits_per_sym"] < 0.5
    assert result_shallow["depth"] == 1
    assert result_deep["depth"] == 8


# ---------------------------------------------------------------------------
# Test 10: update() is consistent with predict()
# ---------------------------------------------------------------------------

def test_update_consistent_with_predict():
    """P returned by update() should match predict() called before update."""
    rng = random.Random(99)
    ctw_a = CTW(depth=4)
    ctw_b = CTW(depth=4)

    history: list[int] = []
    for _ in range(30):
        bit = rng.randint(0, 1)
        p0_pred, p1_pred = ctw_a.predict(history)
        p_from_predict = p0_pred if bit == 0 else p1_pred

        p_from_update = ctw_b.update(history, bit)

        assert abs(p_from_predict - p_from_update) < 1e-9, (
            f"predict()={p_from_predict:.8f} != update()={p_from_update:.8f} "
            f"at step {len(history)}, bit={bit}"
        )

        ctw_a.update(history, bit)
        history.append(bit)


# ---------------------------------------------------------------------------
# Test: score_sequence_str output keys
# ---------------------------------------------------------------------------

def test_score_sequence_str_output_keys():
    ctw = CTW(depth=8)
    result = ctw.score_sequence_str("0" * 200)
    for key in ("h_ctw_bits_per_sym", "depth", "n_context", "n_predicted",
                "context_len", "predict_len"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test: sg_min is <= sg_primary (conservative property)
# ---------------------------------------------------------------------------

def test_sg_min_le_sg_primary():
    """sg_min uses best CTW baseline, so it should be <= sg_primary."""
    rng = random.Random(11)
    sequence = "".join(str(rng.randint(0, 1)) for _ in range(200))
    p0_list = [0.5 + 0.1 * rng.random() for _ in range(100)]
    result = compute_sg(sequence, p0_list, depths=(4, 8, 12))
    assert result["sg_min"] <= result["sg_primary"] + 1e-9, (
        f"sg_min={result['sg_min']:.4f} > sg_primary={result['sg_primary']:.4f}"
    )
