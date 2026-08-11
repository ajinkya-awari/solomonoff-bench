"""Generate Wolfram elementary Cellular Automata sequences for the full benchmark dataset.

Produces 8 Wolfram rules × 75 seeds = 600 unique sequences of 200 bits each.

Rules selected for complexity gradient:
  Rule 30  — chaotic, cryptographic quality
  Rule 45  — chaotic, complex boundary behaviour
  Rule 60  — XOR-based, moderate complexity
  Rule 90  — XOR (self-similar / Sierpinski-like)
  Rule 110  — Turing-complete
  Rule 150  — linear, XOR with three cells
  Rule 184  — traffic-flow model (class 2/3)
  Rule 240  — left-shift rule, simple

Each rule generates 75 sequences by varying the initial 200-cell random row (seeded RNG).
The CA is run for 200 additional steps; the concatenated binary string of the *centre cell*
across all 200 steps forms the 200-symbol output sequence.  The initial row is also 200 cells
wide (periodic boundary) so the centre cell has no boundary artefacts.

Output record format matches sequences_mvp.json TM records exactly, with additional fields:
  generator_type : "ca"
  rule           : int — Wolfram elementary rule number
  seed           : int — RNG seed used to generate the initial row
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

SEQUENCE_LENGTH = 200
SEQUENCES_PER_RULE = 75

# 8 Wolfram elementary CA rules spanning a complexity gradient
CA_RULES: list[int] = [30, 45, 60, 90, 110, 150, 184, 240]

# ── lookup table: precompute the 8 neighbourhood patterns for a given rule ──

def _rule_table(rule: int) -> dict[tuple[int, int, int], int]:
    """Return mapping (left, centre, right) → next_bit for a Wolfram rule [0..255]."""
    table: dict[tuple[int, int, int], int] = {}
    for pattern in range(8):
        left   = (pattern >> 2) & 1
        centre = (pattern >> 1) & 1
        right  = (pattern >> 0) & 1
        output_bit = (rule >> pattern) & 1
        table[(left, centre, right)] = output_bit
    return table


def _run_ca(rule: int, initial_row: list[int], steps: int) -> list[int]:
    """Evolve a Wolfram elementary CA for `steps` generations.

    Parameters
    ----------
    rule        : Wolfram rule number 0–255
    initial_row : list of 0/1 integers, length = SEQUENCE_LENGTH (periodic boundary)
    steps       : number of CA steps to run

    Returns
    -------
    List of `steps` centre-cell values (one per generation), 0 or 1.
    """
    table = _rule_table(rule)
    width = len(initial_row)
    centre_idx = width // 2

    row = initial_row[:]
    centre_trace: list[int] = []

    for _ in range(steps):
        new_row: list[int] = []
        for i in range(width):
            left   = row[(i - 1) % width]
            centre = row[i]
            right  = row[(i + 1) % width]
            new_row.append(table[(left, centre, right)])
        row = new_row
        centre_trace.append(row[centre_idx])

    return centre_trace


def generate_ca_sequences(
    base_seed: int = 3000,
    verbose: bool = True,
    existing_sequences: set[str] | None = None,
) -> list[dict]:
    """Generate all 600 CA sequences (8 rules × 75 seeds).

    Parameters
    ----------
    existing_sequences : optional set of sequence strings already used by other
        generators (e.g. TM sequences from sequences_mvp.json).  New sequences
        are deduplicated against this set in addition to within-CA deduplication.

    Returns a list of record dicts compatible with sequences_mvp.json.
    """
    if verbose:
        print(f"Generating {len(CA_RULES) * SEQUENCES_PER_RULE} CA sequences "
              f"({SEQUENCES_PER_RULE}/rule × {len(CA_RULES)} rules) ...")

    all_records: list[dict] = []
    global_seen: set[str] = set(existing_sequences) if existing_sequences else set()

    for rule in CA_RULES:
        rule_records: list[dict] = []
        seen_in_rule: set[str] = set()

        # Seed is offset per rule to ensure independence
        rng = random.Random(base_seed + rule * 10_000)
        index = 0
        max_attempts = SEQUENCES_PER_RULE * 200  # upper bound on seed draws

        attempts = 0
        while len(rule_records) < SEQUENCES_PER_RULE and attempts < max_attempts:
            attempts += 1
            seed = rng.randint(0, 2**32 - 1)

            # Build the initial 200-cell row from this seed
            row_rng = random.Random(seed)
            initial_row = [row_rng.randint(0, 1) for _ in range(SEQUENCE_LENGTH)]

            # Run the CA for SEQUENCE_LENGTH steps; collect centre cell
            trace = _run_ca(rule, initial_row, SEQUENCE_LENGTH)
            seq = "".join(str(b) for b in trace)

            assert len(seq) == SEQUENCE_LENGTH, f"CA trace length {len(seq)} != {SEQUENCE_LENGTH}"

            # Deduplicate within this rule and globally
            if seq in seen_in_rule or seq in global_seen:
                continue

            seen_in_rule.add(seq)
            global_seen.add(seq)

            record: dict = {
                "sequence_id": f"CA{rule}_{index:04d}",
                "generator_type": "ca",
                "rule": rule,
                "seed": seed,
                "sequence": seq,
                "sequence_length": SEQUENCE_LENGTH,
            }
            rule_records.append(record)
            index += 1

        if len(rule_records) < SEQUENCES_PER_RULE:
            raise RuntimeError(
                f"CA rule {rule}: could only collect {len(rule_records)} unique sequences "
                f"after {attempts} attempts (target {SEQUENCES_PER_RULE})."
            )

        if verbose:
            print(f"  Rule {rule:>3d}: {len(rule_records)} sequences collected "
                  f"(attempts={attempts})")

        all_records.extend(rule_records)

    if verbose:
        print(f"CA total: {len(all_records)} sequences")

    return all_records


def validate_ca_records(records: list[dict]) -> None:
    """Assert all invariants on the CA records."""
    assert len(records) == len(CA_RULES) * SEQUENCES_PER_RULE, (
        f"Expected {len(CA_RULES) * SEQUENCES_PER_RULE} CA records, got {len(records)}"
    )
    seen_ids: set[str] = set()
    seen_seqs: set[str] = set()

    for rec in records:
        sid = rec["sequence_id"]
        assert sid not in seen_ids, f"Duplicate sequence_id: {sid}"
        seen_ids.add(sid)

        seq = rec["sequence"]
        assert seq not in seen_seqs, f"Duplicate sequence for {sid}"
        seen_seqs.add(seq)

        assert len(seq) == SEQUENCE_LENGTH, (
            f"{sid}: length {len(seq)} != {SEQUENCE_LENGTH}"
        )
        assert all(c in "01" for c in seq), f"{sid}: non-binary character"
        assert rec["generator_type"] == "ca"
        assert rec["rule"] in CA_RULES, f"{sid}: unknown rule {rec['rule']}"
        assert rec["sequence_length"] == SEQUENCE_LENGTH

    # Per-rule counts
    for rule in CA_RULES:
        rule_recs = [r for r in records if r["rule"] == rule]
        assert len(rule_recs) == SEQUENCES_PER_RULE, (
            f"Rule {rule}: expected {SEQUENCES_PER_RULE}, got {len(rule_recs)}"
        )

    print("CA validation passed.")


if __name__ == "__main__":
    records = generate_ca_sequences(verbose=True)
    validate_ca_records(records)
