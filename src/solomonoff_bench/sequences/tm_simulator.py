"""One-tape binary output-transducer Turing Machine simulator.

Convention: the written symbol is appended to the output stream at every
transition. Output is an ASCII string of '0'/'1' characters, never packed
binary. Sequences are exactly 200 symbols or discarded.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

OUTPUT_LENGTH = 200
MAX_TRANSITIONS = 10_000

# Blank symbol on the tape
BLANK = 0

# Direction constants
LEFT = 0
RIGHT = 1


@dataclass
class Transition:
    write: int        # 0 or 1
    direction: int    # LEFT or RIGHT
    next_state: int


@dataclass
class TMProgram:
    """A fully specified TM program with metadata."""
    n_states: int
    transition_table: dict[tuple[int, int], Transition]  # (state, read) -> Transition
    program_bits: int
    seed: int

    def encode(self) -> list[int]:
        """Return the bit encoding used to compute program_bits."""
        bits: list[int] = []
        n_bits_state = math.ceil(math.log2(self.n_states + 1))
        for state in range(self.n_states):
            for symbol in range(2):
                t = self.transition_table[(state, symbol)]
                bits.append(t.write)
                bits.append(t.direction)
                # next_state encoded in n_bits_state bits (big-endian)
                ns_bits = format(t.next_state, f"0{n_bits_state}b")
                bits.extend(int(b) for b in ns_bits)
        return bits


def compute_program_bits(n_states: int) -> int:
    """Bits needed to encode a full n-state, 2-symbol TM under our fixed encoding."""
    n_bits_state = math.ceil(math.log2(n_states + 1))
    # Each rule: 1 (write) + 1 (direction) + n_bits_state (next state)
    # Total rules: n_states × 2 (one per (state, symbol) pair)
    return n_states * 2 * (2 + n_bits_state)


def random_tm_program(n_states: int, rng: random.Random) -> TMProgram:
    """Sample a random TM program uniformly from all n-state, 2-symbol TMs."""
    table: dict[tuple[int, int], Transition] = {}
    for state in range(n_states):
        for symbol in range(2):
            write = rng.randint(0, 1)
            direction = rng.randint(0, 1)
            next_state = rng.randint(0, n_states - 1)
            table[(state, symbol)] = Transition(write, direction, next_state)
    prog_bits = compute_program_bits(n_states)
    return TMProgram(n_states=n_states, transition_table=table,
                     program_bits=prog_bits, seed=0)


@dataclass
class TMResult:
    """Result of running a TM program."""
    sequence: str           # ASCII "0"/"1" string of length OUTPUT_LENGTH, or "" if discarded
    accepted: bool
    step_count: int
    discard_reason: Optional[str] = None
    output_convention: str = "output-transducer: written symbol emitted at each transition"


def run_tm(program: TMProgram) -> TMResult:
    """Run the TM as an output transducer.

    Tape: infinite to both sides, initialized to all zeros.
    Head starts at cell 0, state starts at 0.
    At every transition: append the written symbol to the output string.
    Halt when output reaches OUTPUT_LENGTH symbols, or discard after
    MAX_TRANSITIONS steps.
    """
    tape: dict[int, int] = {}  # sparse tape; missing cells read as BLANK
    head = 0
    state = 0
    output_chars: list[str] = []
    steps = 0

    while len(output_chars) < OUTPUT_LENGTH:
        if steps >= MAX_TRANSITIONS:
            return TMResult(
                sequence="",
                accepted=False,
                step_count=steps,
                discard_reason=f"exceeded {MAX_TRANSITIONS} transitions",
            )

        read_sym = tape.get(head, BLANK)
        if (state, read_sym) not in program.transition_table:
            # Undefined transition — treat as halt without completing output
            return TMResult(
                sequence="",
                accepted=False,
                step_count=steps,
                discard_reason="undefined transition (implicit halt before 200 symbols)",
            )

        t = program.transition_table[(state, read_sym)]
        tape[head] = t.write
        output_chars.append(str(t.write))   # transducer: emit written symbol
        head += 1 if t.direction == RIGHT else -1
        state = t.next_state
        steps += 1

    return TMResult(
        sequence="".join(output_chars),
        accepted=True,
        step_count=steps,
    )
