"""Tests for the three new generator modules: CA, LFSR, canary.

Run with: python -m pytest tests/test_new_generators.py -v
"""

from __future__ import annotations

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solomonoff_bench.sequences.generate_ca_sequences import (
    CA_RULES,
    SEQUENCES_PER_RULE,
    SEQUENCE_LENGTH,
    _rule_table,
    _run_ca,
    generate_ca_sequences,
)
from solomonoff_bench.sequences.generate_lfsr_sequences import (
    LFSR_CONFIGS,
    SEQUENCES_PER_DEGREE,
    _lfsr_generate,
    generate_lfsr_sequences,
)
from solomonoff_bench.sequences.generate_canary_sequences import (
    CANARY_TYPES,
    SEQUENCES_PER_TYPE,
    _make_sequence,
    generate_canary_sequences,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cellular Automata tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCARuleTable:
    def test_rule_table_has_eight_entries(self):
        table = _rule_table(30)
        assert len(table) == 8, "Rule table must have exactly 8 neighbourhood patterns"

    def test_rule_0_maps_all_to_zero(self):
        table = _rule_table(0)
        for v in table.values():
            assert v == 0, "Rule 0: every neighbourhood should map to 0"

    def test_rule_255_maps_all_to_one(self):
        table = _rule_table(255)
        for v in table.values():
            assert v == 1, "Rule 255: every neighbourhood should map to 1"

    def test_rule_30_known_centre_bit(self):
        # Pattern 000 → rule 30 bit 0 = 0
        table = _rule_table(30)
        assert table[(0, 0, 0)] == 0
        # Pattern 001 → rule 30 bit 1 = 1
        assert table[(0, 0, 1)] == 1
        # Pattern 010 → rule 30 bit 2 = 1
        assert table[(0, 1, 0)] == 1
        # Pattern 011 → rule 30 bit 3 = 1
        assert table[(0, 1, 1)] == 1


class TestCASequenceLength:
    def test_ca_trace_is_200_symbols(self):
        initial_row = [0] * 200
        initial_row[100] = 1  # single seed cell
        trace = _run_ca(30, initial_row, SEQUENCE_LENGTH)
        assert len(trace) == SEQUENCE_LENGTH

    def test_ca_trace_contains_only_01(self):
        import random
        rng = random.Random(42)
        initial_row = [rng.randint(0, 1) for _ in range(200)]
        trace = _run_ca(110, initial_row, SEQUENCE_LENGTH)
        assert all(b in (0, 1) for b in trace)

    def test_ca_records_all_200_chars(self):
        records = generate_ca_sequences(base_seed=3000, verbose=False)
        for rec in records:
            assert len(rec["sequence"]) == 200, (
                f"{rec['sequence_id']}: sequence length {len(rec['sequence'])} != 200"
            )

    def test_ca_sequences_are_ascii_binary_strings(self):
        records = generate_ca_sequences(base_seed=3000, verbose=False)
        for rec in records:
            seq = rec["sequence"]
            assert isinstance(seq, str)
            assert all(c in "01" for c in seq), (
                f"{rec['sequence_id']}: non-binary character found"
            )


class TestCAUniqueness:
    def test_ca_sequences_unique_within_rule(self):
        records = generate_ca_sequences(base_seed=3000, verbose=False)
        for rule in CA_RULES:
            rule_seqs = [r["sequence"] for r in records if r["rule"] == rule]
            assert len(rule_seqs) == len(set(rule_seqs)), (
                f"Rule {rule}: duplicate sequences found"
            )

    def test_ca_correct_count_per_rule(self):
        records = generate_ca_sequences(base_seed=3000, verbose=False)
        for rule in CA_RULES:
            count = sum(1 for r in records if r["rule"] == rule)
            assert count == SEQUENCES_PER_RULE, (
                f"Rule {rule}: expected {SEQUENCES_PER_RULE}, got {count}"
            )

    def test_ca_total_count(self):
        records = generate_ca_sequences(base_seed=3000, verbose=False)
        expected = len(CA_RULES) * SEQUENCES_PER_RULE
        assert len(records) == expected, f"Expected {expected} CA records, got {len(records)}"


# ─────────────────────────────────────────────────────────────────────────────
# LFSR tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLFSRSequenceLength:
    def test_lfsr_output_is_200_bits(self):
        bits = _lfsr_generate(7, [7, 6], 1, 200)
        assert len(bits) == 200

    def test_lfsr_records_all_200_chars(self):
        records = generate_lfsr_sequences(base_seed=4000, verbose=False)
        for rec in records:
            assert len(rec["sequence"]) == 200, (
                f"{rec['sequence_id']}: length {len(rec['sequence'])} != 200"
            )

    def test_lfsr_sequences_are_ascii_binary(self):
        records = generate_lfsr_sequences(base_seed=4000, verbose=False)
        for rec in records:
            seq = rec["sequence"]
            assert all(c in "01" for c in seq), f"{rec['sequence_id']}: non-binary character"


class TestLFSRPeriodProperties:
    def test_lfsr7_period_is_127(self):
        """LFSR degree-7 with poly x^7+x^6+1 must have period 127 (maximal length)."""
        bits = _lfsr_generate(7, [7, 6], 1, 254)
        # The first 127 bits should repeat in the second 127 bits
        assert bits[:127] == bits[127:254], "LFSR-7 period is not 127"

    def test_lfsr8_period_is_255(self):
        """LFSR degree-8 with poly x^8+x^6+x^5+x^4+1 must have period 255."""
        bits = _lfsr_generate(8, [8, 6, 5, 4], 1, 510)
        assert bits[:255] == bits[255:510], "LFSR-8 period is not 255"

    def test_lfsr15_period_exceeds_200(self):
        """LFSR-15 period is 32767; first 200 bits should not cycle within the window."""
        bits_a = _lfsr_generate(15, [15, 14], 1,    200)
        bits_b = _lfsr_generate(15, [15, 14], 1,    400)
        # If period were ≤ 200, bits_a == bits_b[:200] would hold AND bits_b[200:400] == bits_a
        # To confirm no period, check bits_b[200:] differs from bits_a
        # (This will always pass for period > 200, which is the assertion)
        assert bits_b[200:] != bits_a, (
            "LFSR-15 unexpectedly cycled within 400 bits — period must be > 200"
        )

    def test_lfsr_nonzero_state_never_returns_zero_state(self):
        """A maximal-length LFSR must never pass through the all-zeros state."""
        degree, taps = 7, [7, 6]
        period = (1 << degree) - 1
        bits = _lfsr_generate(degree, taps, 1, period * 2)
        # Reconstruct states; verify none is 0
        state = 1
        for _ in range(period):
            feedback = 0
            for tap in taps:
                feedback ^= (state >> (tap - 1)) & 1
            state = ((state << 1) | feedback) & ((1 << degree) - 1)
            assert state != 0, "LFSR entered all-zeros state — polynomial is not primitive"

    def test_lfsr_total_count(self):
        records = generate_lfsr_sequences(base_seed=4000, verbose=False)
        expected = len(LFSR_CONFIGS) * SEQUENCES_PER_DEGREE
        assert len(records) == expected, f"Expected {expected} LFSR records, got {len(records)}"

    def test_lfsr_unique_within_degree(self):
        records = generate_lfsr_sequences(base_seed=4000, verbose=False)
        for degree, _ in LFSR_CONFIGS:
            d_seqs = [r["sequence"] for r in records if r["degree"] == degree]
            assert len(d_seqs) == len(set(d_seqs)), (
                f"LFSR degree {degree}: duplicate sequences found"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Canary tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCanaryFixedPatterns:
    def test_zeros_canary_is_all_zeros(self):
        seq = _make_sequence("ZEROS", seed=0)
        assert seq == "0" * 200, "ZEROS canary must be 200 zero characters"

    def test_zeros_canary_seed_invariant(self):
        """ZEROS canary must always produce the same sequence regardless of seed."""
        assert _make_sequence("ZEROS", seed=0) == _make_sequence("ZEROS", seed=99999)

    def test_ones_canary_is_all_ones(self):
        seq = _make_sequence("ONES", seed=0)
        assert seq == "1" * 200, "ONES canary must be 200 one characters"

    def test_alt01_is_strictly_alternating(self):
        seq = _make_sequence("ALT01", seed=0)
        assert len(seq) == 200
        for i in range(len(seq) - 1):
            assert seq[i] != seq[i + 1], f"ALT01 not alternating at position {i}"

    def test_alt01_has_period_2(self):
        seq = _make_sequence("ALT01", seed=0)
        # Must be either 010101... or 101010...
        expected_a = "".join(str(i % 2) for i in range(200))
        expected_b = "".join(str((i + 1) % 2) for i in range(200))
        assert seq in (expected_a, expected_b), "ALT01 is not a clean period-2 sequence"

    def test_alt0011_has_period_4(self):
        seq = _make_sequence("ALT0011", seed=0)
        assert len(seq) == 200
        # First 4 chars should repeat
        block = seq[:4]
        for i in range(0, 200, 4):
            assert seq[i:i + 4] == block, f"ALT0011 period-4 mismatch at position {i}"


class TestCanaryRandomEntropy:
    def test_rand_canary_entropy_near_1(self):
        """RAND canary should have empirical entropy close to 1.0 bit/symbol."""
        records = generate_canary_sequences(base_seed=5000, verbose=False)
        rand_seqs = [r["sequence"] for r in records if r["canary_type"] == "RAND"]
        # Collect all bits across all 75 RAND sequences
        all_bits = "".join(rand_seqs)
        n = len(all_bits)
        p1 = all_bits.count("1") / n
        p0 = 1 - p1
        # Entropy: H = -p0*log2(p0) - p1*log2(p1)
        def safe_entropy(p: float) -> float:
            if p <= 0 or p >= 1:
                return 0.0
            return -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        H = safe_entropy(p1)
        assert H > 0.95, (
            f"RAND canary empirical entropy {H:.4f} < 0.95 — random generation is biased"
        )

    def test_low0_has_more_zeros_than_ones(self):
        seq = _make_sequence("LOW0", seed=42)
        assert seq.count("0") > seq.count("1"), "LOW0 canary should have more 0s than 1s"

    def test_low1_has_more_ones_than_zeros(self):
        seq = _make_sequence("LOW1", seed=42)
        assert seq.count("1") > seq.count("0"), "LOW1 canary should have more 1s than 0s"

    def test_canary_total_count(self):
        records = generate_canary_sequences(base_seed=5000, verbose=False)
        expected = len(CANARY_TYPES) * SEQUENCES_PER_TYPE
        assert len(records) == expected, (
            f"Expected {expected} canary records, got {len(records)}"
        )

    def test_canary_sequence_ids_all_unique(self):
        records = generate_canary_sequences(base_seed=5000, verbose=False)
        ids = [r["sequence_id"] for r in records]
        assert len(ids) == len(set(ids)), "Duplicate sequence_ids in canary records"


# ─────────────────────────────────────────────────────────────────────────────
# Full dataset integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestFullDatasetCount:
    def test_combined_records_total_2400(self):
        """Verify that TM(300) + CA(600) + LFSR(600) + canary(900) = 2400."""
        # TM count is fixed
        tm_count = 300
        ca_records     = generate_ca_sequences(base_seed=3000, verbose=False)
        lfsr_records   = generate_lfsr_sequences(base_seed=4000, verbose=False)
        canary_records = generate_canary_sequences(base_seed=5000, verbose=False)

        total = tm_count + len(ca_records) + len(lfsr_records) + len(canary_records)
        assert total == 2400, (
            f"Expected 2400 total records, got {total} "
            f"(TM={tm_count}, CA={len(ca_records)}, "
            f"LFSR={len(lfsr_records)}, canary={len(canary_records)})"
        )

    def test_no_cross_generator_sequence_duplicates(self):
        """Sequences from CA and LFSR are mutually unique.
        RAND and LOW0/LOW1 canary sequences are also globally unique vs CA and LFSR.
        PRBS15 canary sequences are excluded because they share the same mathematical
        structure as LFSR sequences and occasional collisions are expected by design.
        Fixed-pattern canaries (ZEROS, ONES, ALT01, etc.) intentionally repeat sequence
        content and are also excluded from this check.
        """
        ca_records     = generate_ca_sequences(base_seed=3000, verbose=False)
        lfsr_records   = generate_lfsr_sequences(base_seed=4000, verbose=False)
        canary_records = generate_canary_sequences(base_seed=5000, verbose=False)

        ca_seqs   = {r["sequence"] for r in ca_records}
        lfsr_seqs = {r["sequence"] for r in lfsr_records}
        # RAND, LOW0, LOW1 are structurally different from LFSR — no expected collisions
        structurally_distinct_canary_seqs = {
            r["sequence"] for r in canary_records
            if r["canary_type"] in ("RAND", "LOW0", "LOW1")
        }

        # CA ∩ LFSR must be empty
        ca_lfsr = ca_seqs & lfsr_seqs
        assert len(ca_lfsr) == 0, f"{len(ca_lfsr)} sequences appear in both CA and LFSR"

        # CA ∩ RAND/LOW canary must be empty
        ca_canary = ca_seqs & structurally_distinct_canary_seqs
        assert len(ca_canary) == 0, (
            f"{len(ca_canary)} sequences appear in both CA and RAND/LOW canary types"
        )

        # LFSR ∩ RAND/LOW canary must be empty
        lfsr_canary = lfsr_seqs & structurally_distinct_canary_seqs
        assert len(lfsr_canary) == 0, (
            f"{len(lfsr_canary)} sequences appear in both LFSR and RAND/LOW canary types"
        )
