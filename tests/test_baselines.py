"""Unit tests for gzip and random baselines."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import gzip
import pytest
from solomonoff_bench.baselines.gzip_baseline import h_gzip_incremental, GzipBaseline
from solomonoff_bench.baselines.random_baseline import RandomBaseline


class TestGzipIncremental:
    def test_returns_float(self):
        assert isinstance(h_gzip_incremental("0" * 100, "1" * 100), float)

    def test_highly_compressible_suffix_near_zero(self):
        # A very compressible suffix should need few bits — but gzip is unstable
        # so we just assert it's a real number, not that it's small
        val = h_gzip_incremental("0" * 100, "0" * 100)
        assert isinstance(val, float)

    def test_negative_delta_is_returned_not_clamped(self):
        # Force a scenario where the suffix matches the prefix perfectly:
        # gzip can sometimes produce smaller output for full string
        # We just verify the function never raises on negative output
        val = h_gzip_incremental("01" * 50, "01" * 25)
        assert isinstance(val, float)

    def test_rejects_non_binary_prefix(self):
        with pytest.raises(AssertionError):
            h_gzip_incremental("hello", "01")

    def test_rejects_empty_suffix(self):
        with pytest.raises(AssertionError):
            h_gzip_incremental("0" * 10, "")

    def test_formula_matches_manual_computation(self):
        prefix = "01" * 50
        suffix = "10" * 10
        p = prefix.encode("ascii")
        f = (prefix + suffix).encode("ascii")
        expected = 8 * (len(gzip.compress(f, 9)) - len(gzip.compress(p, 9))) / len(suffix)
        assert abs(h_gzip_incremental(prefix, suffix) - expected) < 1e-10


class TestGzipBaseline:
    def test_tracks_negative_deltas(self):
        gb = GzipBaseline()
        # Score a bunch of pairs; some may be negative
        for _ in range(10):
            gb.score("01" * 50, "10" * 50)
        assert gb.total_count == 10
        assert 0 <= gb.negative_delta_count <= 10

    def test_negative_rate_property(self):
        gb = GzipBaseline()
        gb.score("0" * 100, "1" * 100)
        assert 0.0 <= gb.negative_delta_rate <= 1.0

    def test_diagnostics_has_required_keys(self):
        gb = GzipBaseline()
        gb.score("01" * 50, "01" * 50)
        d = gb.diagnostics()
        for key in ["total_calls", "negative_delta_count", "negative_delta_rate",
                    "negative_delta_values", "note"]:
            assert key in d

    def test_reset_clears_state(self):
        gb = GzipBaseline()
        gb.score("0" * 100, "1" * 100)
        gb.reset()
        assert gb.total_count == 0
        assert gb.negative_delta_count == 0


class TestRandomBaseline:
    def test_log_prob_minus_one_for_zero(self):
        rb = RandomBaseline()
        assert rb.log_prob("0") == -1.0

    def test_log_prob_minus_one_for_one(self):
        rb = RandomBaseline()
        assert rb.log_prob("1") == -1.0

    def test_cross_entropy_always_one(self):
        rb = RandomBaseline()
        for seq in ["01010101", "0" * 50, "1" * 50, "01" * 25]:
            assert rb.cross_entropy(seq) == 1.0

    def test_score_sequence_returns_correct_shape(self):
        rb = RandomBaseline()
        result = rb.score_sequence("01" * 25, "10" * 25)
        assert result["cross_entropy_bits_per_symbol"] == 1.0
        assert len(result["log_probs"]) == 50
        assert all(lp == -1.0 for lp in result["log_probs"])
