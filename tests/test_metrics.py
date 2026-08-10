"""Unit tests for excess_loss (EL_gzip) metric."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math
import pytest
from solomonoff_bench.metrics.excess_loss import cross_entropy_from_probs, compute_el_gzip


class TestCrossEntropyFromProbs:
    def test_perfect_predictor_zero_entropy(self):
        # Always predicts the true bit with P=1.0
        bits = "01010101"
        p0s = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        h = cross_entropy_from_probs(bits, p0s)
        assert h < 1e-6

    def test_random_predictor_entropy_one(self):
        bits = "01010101"
        p0s = [0.5] * 8
        h = cross_entropy_from_probs(bits, p0s)
        assert abs(h - 1.0) < 1e-6

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            cross_entropy_from_probs("010", [0.5, 0.5])

    def test_near_zero_prob_clamped_no_inf(self):
        bits = "0"
        p0s = [0.0]  # P(0)=0 but true bit is 0 — normally log(0)
        h = cross_entropy_from_probs(bits, p0s)
        assert math.isfinite(h)
        assert h > 0


class TestComputeElGzip:
    def _make_seq(self, prefix_char="0", suffix_char="1"):
        return prefix_char * 100 + suffix_char * 100

    def test_returns_required_keys(self):
        seq = self._make_seq()
        result = compute_el_gzip(seq, [0.5] * 100)
        for key in ["h_model_bits_per_sym", "h_gzip_bits_per_sym",
                    "h_gzip_is_negative", "el_gzip", "context_len", "predict_len"]:
            assert key in result

    def test_random_predictor_h_model_is_one(self):
        seq = self._make_seq()
        result = compute_el_gzip(seq, [0.5] * 100)
        assert abs(result["h_model_bits_per_sym"] - 1.0) < 1e-5

    def test_h_gzip_is_negative_flag_correct(self):
        seq = self._make_seq()
        result = compute_el_gzip(seq, [0.5] * 100)
        h_gzip = result["h_gzip_bits_per_sym"]
        assert result["h_gzip_is_negative"] == (h_gzip < 0)

    def test_wrong_sequence_length_raises(self):
        with pytest.raises(AssertionError):
            compute_el_gzip("0" * 150, [0.5] * 100)

    def test_el_gzip_equals_h_model_minus_h_gzip(self):
        seq = "01" * 50 + "10" * 50  # exactly 200 chars
        result = compute_el_gzip(seq, [0.5] * 100)
        expected = result["h_model_bits_per_sym"] - result["h_gzip_bits_per_sym"]
        assert abs(result["el_gzip"] - expected) < 1e-10
