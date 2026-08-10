"""Uniform random baseline: always predicts P(0) = P(1) = 0.5.

H(random, x) = 1.0 bits/symbol always.
Any model with H > 1.0 is worse than random on that sequence.
"""

from __future__ import annotations

import math


LOG2_2 = math.log(2) / math.log(2)  # = 1.0


class RandomBaseline:
    """Uniform random predictor. Provides log-probability interface."""

    def log_prob(self, bit: str) -> float:
        """Return log2 P(bit) = log2(0.5) = -1.0 for any bit."""
        assert bit in ("0", "1"), f"Expected '0' or '1', got {bit!r}"
        return -1.0

    def cross_entropy(self, sequence: str) -> float:
        """Return cross-entropy in bits/symbol = 1.0 always."""
        assert all(c in "01" for c in sequence), "sequence must be '0'/'1' ASCII"
        return 1.0

    def score_sequence(self, prefix: str, suffix: str) -> dict:
        """Return per-symbol log-probs and cross-entropy for the suffix."""
        assert all(c in "01" for c in suffix)
        log_probs = [-1.0] * len(suffix)
        return {
            "log_probs": log_probs,
            "cross_entropy_bits_per_symbol": 1.0,
            "n_symbols": len(suffix),
        }
