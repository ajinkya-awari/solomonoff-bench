"""Generate Linear Feedback Shift Register (LFSR) sequences for the full benchmark dataset.

Produces 8 LFSR degrees × 75 seeds = 600 unique sequences of 200 bits each.

Degrees and their standard maximal-length (primitive) feedback polynomials:
  Degree  7 — poly x^7 + x^6 + 1                   taps [7, 6]        period 127
  Degree  8 — poly x^8 + x^6 + x^5 + x^4 + 1      taps [8, 6, 5, 4]  period 255
  Degree  9 — poly x^9 + x^5 + 1                   taps [9, 5]        period 511
  Degree 11 — poly x^11 + x^9 + 1                  taps [11, 9]       period 2047
  Degree 13 — poly x^13 + x^12 + x^11 + x^8 + 1   taps [13,12,11,8]  period 8191
  Degree 15 — poly x^15 + x^14 + 1                 taps [15, 14]      period 32767
  Degree 17 — poly x^17 + x^14 + 1                 taps [17, 14]      period 131071
  Degree 23 — poly x^23 + x^18 + 1                 taps [23, 18]      period 8388607

All polynomials are well-known primitive polynomials over GF(2).  For degree 7 the
period is 127 bits; since 127 ≥ 75 unique start states are available and tiling
produces distinct 200-bit strings for distinct start offsets, we have ≥ 75 unique
sequences.  For degree ≥ 8 the period exceeds 200 bits so every non-zero seed
immediately produces a unique 200-bit prefix.

Output format matches sequences_mvp.json exactly, with additional fields:
  generator_type : "lfsr"
  degree         : LFSR degree
  taps           : list of tap positions (feedback polynomial)
  seed           : integer seed — used as the LFSR initial state (non-zero, mod 2^degree)
"""

from __future__ import annotations

import random
from pathlib import Path

SEQUENCE_LENGTH = 200
SEQUENCES_PER_DEGREE = 75

# (degree, taps) — all use standard primitive polynomials over GF(2)
# Tap positions use the convention: feedback = XOR of bits at positions taps[i]-1 (0-indexed)
# Degree 5 removed: period 31 < 75 unique sequences required per degree.
LFSR_CONFIGS: list[tuple[int, list[int]]] = [
    (7,  [7, 6]),
    (8,  [8, 6, 5, 4]),
    (9,  [9, 5]),
    (11, [11, 9]),
    (13, [13, 12, 11, 8]),
    (15, [15, 14]),
    (17, [17, 14]),
    (23, [23, 18]),
]


def _lfsr_generate(degree: int, taps: list[int], initial_state: int, n_bits: int) -> list[int]:
    """Run a Galois LFSR for `n_bits` steps, returning the output bit stream.

    The LFSR is a Fibonacci LFSR (simple shift register with XOR feedback).
    Tap positions are 1-indexed (matching the polynomial notation).

    Parameters
    ----------
    degree        : number of stages
    taps          : list of tap positions (1-indexed, must include degree)
    initial_state : initial state as a non-zero integer (bits 0..degree-1)
    n_bits        : number of output bits to produce

    Returns
    -------
    List of `n_bits` integers (0 or 1), MSB of the shift register each step.
    """
    if initial_state == 0:
        raise ValueError("LFSR initial state must be non-zero")

    state = initial_state & ((1 << degree) - 1)
    if state == 0:
        state = 1  # safety: clamp to non-zero

    output: list[int] = []
    for _ in range(n_bits):
        # Output the MSB (bit at position degree-1, 0-indexed)
        bit_out = (state >> (degree - 1)) & 1
        output.append(bit_out)

        # Compute feedback: XOR of bits at tap positions (1-indexed)
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (tap - 1)) & 1

        # Shift left, mask to degree bits, inject feedback at LSB
        state = ((state << 1) | feedback) & ((1 << degree) - 1)

    return output


def generate_lfsr_sequences(
    base_seed: int = 4000,
    verbose: bool = True,
    existing_sequences: set[str] | None = None,
) -> list[dict]:
    """Generate all 600 LFSR sequences (8 degrees × 75 seeds).

    Parameters
    ----------
    existing_sequences : optional set of sequence strings already used by other
        generators (e.g. TM or CA sequences).  New sequences are deduplicated
        against this set in addition to within-LFSR deduplication.

    Returns a list of record dicts compatible with sequences_mvp.json.
    """
    if verbose:
        print(f"Generating {len(LFSR_CONFIGS) * SEQUENCES_PER_DEGREE} LFSR sequences "
              f"({SEQUENCES_PER_DEGREE}/degree × {len(LFSR_CONFIGS)} degrees) ...")

    all_records: list[dict] = []
    global_seen: set[str] = set(existing_sequences) if existing_sequences else set()

    for degree, taps in LFSR_CONFIGS:
        degree_records: list[dict] = []
        seen_in_degree: set[str] = set()

        rng = random.Random(base_seed + degree * 10_000)
        max_state = (1 << degree) - 1  # 2^degree - 1 non-zero states available
        index = 0
        max_attempts = SEQUENCES_PER_DEGREE * 200

        attempts = 0
        while len(degree_records) < SEQUENCES_PER_DEGREE and attempts < max_attempts:
            attempts += 1
            seed = rng.randint(1, max_state)  # non-zero LFSR state

            bits = _lfsr_generate(degree, taps, seed, SEQUENCE_LENGTH)
            seq = "".join(str(b) for b in bits)

            assert len(seq) == SEQUENCE_LENGTH

            if seq in seen_in_degree or seq in global_seen:
                continue

            seen_in_degree.add(seq)
            global_seen.add(seq)

            record: dict = {
                "sequence_id": f"LFSR{degree}_{index:04d}",
                "generator_type": "lfsr",
                "degree": degree,
                "taps": taps,
                "seed": seed,
                "sequence": seq,
                "sequence_length": SEQUENCE_LENGTH,
            }
            degree_records.append(record)
            index += 1

        if len(degree_records) < SEQUENCES_PER_DEGREE:
            raise RuntimeError(
                f"LFSR degree {degree}: could only collect {len(degree_records)} unique sequences "
                f"after {attempts} attempts (target {SEQUENCES_PER_DEGREE})."
            )

        if verbose:
            print(f"  Degree {degree:>2d} (taps={taps}): {len(degree_records)} sequences "
                  f"(attempts={attempts})")

        all_records.extend(degree_records)

    if verbose:
        print(f"LFSR total: {len(all_records)} sequences")

    return all_records


def validate_lfsr_records(records: list[dict]) -> None:
    """Assert all invariants on the LFSR records."""
    degrees = [d for d, _ in LFSR_CONFIGS]
    assert len(records) == len(LFSR_CONFIGS) * SEQUENCES_PER_DEGREE, (
        f"Expected {len(LFSR_CONFIGS) * SEQUENCES_PER_DEGREE} LFSR records, got {len(records)}"
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

        assert len(seq) == SEQUENCE_LENGTH, f"{sid}: length {len(seq)} != {SEQUENCE_LENGTH}"
        assert all(c in "01" for c in seq), f"{sid}: non-binary character"
        assert rec["generator_type"] == "lfsr"
        assert rec["degree"] in degrees, f"{sid}: unknown degree {rec['degree']}"
        assert rec["sequence_length"] == SEQUENCE_LENGTH

    for degree, _ in LFSR_CONFIGS:
        d_recs = [r for r in records if r["degree"] == degree]
        assert len(d_recs) == SEQUENCES_PER_DEGREE, (
            f"Degree {degree}: expected {SEQUENCES_PER_DEGREE}, got {len(d_recs)}"
        )

    print("LFSR validation passed.")


if __name__ == "__main__":
    records = generate_lfsr_sequences(verbose=True)
    validate_lfsr_records(records)
