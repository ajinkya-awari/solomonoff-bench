"""Unit tests for the TM output-transducer simulator.

All tests must pass with: python -m pytest tests/test_tm_simulator.py -v
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solomonoff_bench.sequences.tm_simulator import (
    OUTPUT_LENGTH,
    MAX_TRANSITIONS,
    Transition,
    TMProgram,
    TMResult,
    compute_program_bits,
    random_tm_program,
    run_tm,
)


# ---------------------------------------------------------------------------
# Helper: build a TM that cycles deterministically
# ---------------------------------------------------------------------------

def _always_write_and_go_right(n_states: int, write: int = 1) -> TMProgram:
    """TM that always writes `write`, moves right, stays in state 0."""
    table = {}
    for state in range(n_states):
        for sym in range(2):
            table[(state, sym)] = Transition(write=write, direction=1, next_state=0)
    return TMProgram(
        n_states=n_states,
        transition_table=table,
        program_bits=compute_program_bits(n_states),
        seed=0,
    )


def _alternating_tm() -> TMProgram:
    """2-state TM: state 0 writes 0, state 1 writes 1, alternates states."""
    table = {
        (0, 0): Transition(write=0, direction=1, next_state=1),
        (0, 1): Transition(write=0, direction=1, next_state=1),
        (1, 0): Transition(write=1, direction=1, next_state=0),
        (1, 1): Transition(write=1, direction=1, next_state=0),
    }
    return TMProgram(n_states=2, transition_table=table,
                     program_bits=compute_program_bits(2), seed=0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAcceptedSequenceLength:
    def test_always_write_1_produces_200_symbols(self):
        prog = _always_write_and_go_right(n_states=2, write=1)
        result = run_tm(prog)
        assert result.accepted, "TM should be accepted"
        assert len(result.sequence) == OUTPUT_LENGTH, (
            f"Expected {OUTPUT_LENGTH} symbols, got {len(result.sequence)}"
        )

    def test_always_write_0_produces_200_symbols(self):
        prog = _always_write_and_go_right(n_states=2, write=0)
        result = run_tm(prog)
        assert result.accepted
        assert len(result.sequence) == OUTPUT_LENGTH

    def test_alternating_produces_200_symbols(self):
        prog = _alternating_tm()
        result = run_tm(prog)
        assert result.accepted
        assert len(result.sequence) == OUTPUT_LENGTH

    def test_random_accepted_tms_all_200_symbols(self):
        """Sample random TMs until we collect 20 accepted ones; all must be 200 symbols."""
        rng = random.Random(42)
        accepted = []
        for _ in range(2000):
            prog = random_tm_program(n_states=3, rng=rng)
            prog.seed = 42
            result = run_tm(prog)
            if result.accepted:
                accepted.append(result)
            if len(accepted) >= 20:
                break
        assert len(accepted) >= 5, "Expected to find at least 5 accepted TMs in 2000 tries"
        for r in accepted:
            assert len(r.sequence) == OUTPUT_LENGTH, (
                f"Accepted sequence length {len(r.sequence)} != {OUTPUT_LENGTH}"
            )


class TestAsciiCharacters:
    def test_all_chars_are_0_or_1_ascii(self):
        prog = _alternating_tm()
        result = run_tm(prog)
        assert result.accepted
        for i, ch in enumerate(result.sequence):
            assert ch in ("0", "1"), f"Position {i}: unexpected char {ch!r}"

    def test_sequence_is_str_not_bytes(self):
        prog = _always_write_and_go_right(2, write=1)
        result = run_tm(prog)
        assert isinstance(result.sequence, str), "Sequence must be a Python str"

    def test_sequence_is_not_packed_binary(self):
        prog = _always_write_and_go_right(2, write=1)
        result = run_tm(prog)
        # Packed binary would have len = 25 bytes (200 bits / 8), not 200 chars
        assert len(result.sequence) == 200
        assert all(c in "01" for c in result.sequence)


class TestOutputTransducerConvention:
    def test_alternating_emits_on_every_transition(self):
        """Verify the transducer emits the written symbol, not the read symbol."""
        prog = _alternating_tm()
        result = run_tm(prog)
        # State 0 always writes 0; state 1 always writes 1; alternates
        # Expected: 0,1,0,1,0,1,...
        expected = "".join("0" if i % 2 == 0 else "1" for i in range(OUTPUT_LENGTH))
        assert result.sequence == expected, (
            f"Transducer output mismatch.\n"
            f"Expected: {expected[:20]}...\n"
            f"Got:      {result.sequence[:20]}..."
        )

    def test_always_write_1_output_all_ones(self):
        """If every transition writes 1, output must be all 1s."""
        prog = _always_write_and_go_right(2, write=1)
        result = run_tm(prog)
        assert result.sequence == "1" * OUTPUT_LENGTH

    def test_always_write_0_output_all_zeros(self):
        prog = _always_write_and_go_right(2, write=0)
        result = run_tm(prog)
        assert result.sequence == "0" * OUTPUT_LENGTH

    def test_step_count_equals_output_length_on_simple_tms(self):
        """For TMs that always complete exactly, step_count == OUTPUT_LENGTH."""
        prog = _always_write_and_go_right(2, write=1)
        result = run_tm(prog)
        assert result.step_count == OUTPUT_LENGTH


class TestMaxTransitionsDiscard:
    def test_non_productive_tm_returns_none_result(self):
        """A TM that loops without writing enough symbols should be discarded."""
        # 1-state TM that writes BLANK (0), doesn't move (direction=0), stays in state 0
        # — it will loop forever on cell 0 without advancing output past 200 symbols
        # Actually: it writes 0 each step → it DOES emit output. Let's make it emit but loop.
        # Instead, build a TM that goes left forever (stays on same cell) writing the same value.
        # Tape writes write=1 each time but direction=LEFT keeps head at -1,-2,...
        # Output will accumulate — let's use a stationary TM that emits but never reaches 200.
        # Easiest: make a TM where the output path is blocked (undefined transition).
        # Better: use a 1-state TM that alternates R/L without producing enough output.
        # The simplest: a 2-state TM with a transition loop that goes RIGHT then LEFT forever
        # in state 0 (never reaching output of 200).
        # Actually the easiest way: patch MAX_TRANSITIONS via monkeypatching isn't clean.
        # Instead, verify the discard path exists via the "implicit halt" path.

        # Build a TM where some transitions are missing (partial table) — will halt early
        # Actually all our TMs have complete tables (random_tm_program fills all entries).
        # Let's build a partial TM explicitly:
        table = {
            (0, 0): Transition(write=0, direction=1, next_state=0),
            (0, 1): Transition(write=1, direction=1, next_state=0),
            # State 1 transitions intentionally missing → undefined halt
        }
        prog = TMProgram(n_states=2, transition_table=table,
                         program_bits=compute_program_bits(2), seed=0)
        # This TM is fine — it never enters state 1. It will write and produce 200 symbols.
        # For the actual discard-at-max-transitions test, we need to patch the module.
        # Use importlib to access the module variable.
        import solomonoff_bench.sequences.tm_simulator as mod
        original = mod.MAX_TRANSITIONS
        mod.MAX_TRANSITIONS = 5  # force early discard
        try:
            result = run_tm(prog)
            assert not result.accepted, "Should be discarded when MAX_TRANSITIONS exceeded"
            assert result.sequence == ""
            assert result.discard_reason is not None
            assert "transitions" in result.discard_reason
        finally:
            mod.MAX_TRANSITIONS = original

    def test_discarded_tm_has_empty_sequence(self):
        import solomonoff_bench.sequences.tm_simulator as mod
        original = mod.MAX_TRANSITIONS
        mod.MAX_TRANSITIONS = 3
        try:
            prog = _always_write_and_go_right(2, write=1)
            # With limit=3, only 3 symbols emitted — not 200 → discard
            result = run_tm(prog)
            assert not result.accepted
            assert result.sequence == ""
        finally:
            mod.MAX_TRANSITIONS = original

    def test_discarded_tm_step_count_equals_limit(self):
        import solomonoff_bench.sequences.tm_simulator as mod
        original = mod.MAX_TRANSITIONS
        limit = 10
        mod.MAX_TRANSITIONS = limit
        try:
            prog = _always_write_and_go_right(2, write=1)
            result = run_tm(prog)
            assert not result.accepted
            assert result.step_count == limit
        finally:
            mod.MAX_TRANSITIONS = original


class TestProgramBits:
    def test_2_state_program_bits(self):
        # n_states=2, n_bits_state=ceil(log2(3))=2
        # bits = 2 * 2 * (2 + 2) = 16
        assert compute_program_bits(2) == 16

    def test_3_state_program_bits(self):
        # n_bits_state=ceil(log2(4))=2
        # bits = 3 * 2 * (2 + 2) = 24
        assert compute_program_bits(3) == 24

    def test_4_state_program_bits(self):
        # n_bits_state=ceil(log2(5))=3
        # bits = 4 * 2 * (2 + 3) = 40
        assert compute_program_bits(4) == 40

    def test_5_state_program_bits(self):
        # n_bits_state=ceil(log2(6))=3
        # bits = 5 * 2 * (2 + 3) = 50
        assert compute_program_bits(5) == 50

    def test_random_tm_program_has_correct_bits(self):
        rng = random.Random(0)
        for n in [2, 3, 4, 5]:
            prog = random_tm_program(n, rng)
            assert prog.program_bits == compute_program_bits(n)


class TestOutputConventionField:
    def test_accepted_result_has_convention_string(self):
        prog = _always_write_and_go_right(2)
        result = run_tm(prog)
        assert "output-transducer" in result.output_convention
        assert "written symbol" in result.output_convention
