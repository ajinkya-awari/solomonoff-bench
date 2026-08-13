"""Generate canary control sequences for the full benchmark dataset.

Canary sequences are deterministic controls with known theoretical properties.
The SG benchmark uses them to sanity-check model behaviour:

  - All-zeros / all-ones   : trivially compressible; EL_gzip ≈ 0, SG should be HIGH.
  - Alternating period-2/4 : low-complexity periodic signals.
  - Fair-coin random       : maximally incompressible; SG should be ≈ 0.
  - Low-frequency 0 / 1   : 90 % bias toward one symbol.
  - Sawtooth / PRBS        : periodic ramps and longer-period pseudo-random patterns.

12 canary types × 75 seeds = 900 sequences of 200 bits.

Canary types:
  ZEROS    : all 0s
  ONES     : all 1s
  ALT01    : alternating 01010101... (period 2, 2 unique variants)
  ALT0011  : period-4 pattern 00110011... (4 unique phase variants)
  RAND     : fair-coin random bits (seeded; 75 unique sequences)
  LOW0     : 90 % 0s (random placement of 10 % 1s, seeded; 75 unique)
  LOW1     : 90 % 1s (random placement of 10 % 0s, seeded; 75 unique)
  SAWTOOTH : binary sawtooth — period-8, 8 unique phase variants
  PRBS7    : PRBS from LFSR-7 (period 127, 127 unique start offsets)
  PRBS15   : PRBS from LFSR-15 (period 32767; 75 unique seeds)
  BLOCK    : alternating blocks of 8; phase 0 or 1 (2 unique variants)
  GRAY     : Gray code LSB: period-4 (4 unique phase variants)

Note on sequence uniqueness: fixed-pattern canaries (ZEROS, ONES, etc.) intentionally
produce a small number of distinct sequence strings (e.g. ZEROS has exactly one).
These 75 records per type have unique sequence_ids but may share the same sequence
content.  This is by design — the canary purpose is to test *known-pattern* detection,
not to supply novel sequences.  Only RAND, LOW0, LOW1, PRBS15 produce 75 fully
distinct sequences.

Output format matches sequences_mvp.json exactly, with additional fields:
  generator_type : "canary"
  canary_type    : string identifier
  seed           : integer (used for random canaries; phase offset for periodic ones)
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

SEQUENCE_LENGTH = 200
SEQUENCES_PER_TYPE = 75

CANARY_TYPES: list[str] = [
    "ZEROS",
    "ONES",
    "ALT01",
    "ALT0011",
    "RAND",
    "LOW0",
    "LOW1",
    "SAWTOOTH",
    "PRBS7",
    "PRBS15",
    "BLOCK",
    "GRAY",
]

# Canary types that can produce 75 globally unique sequences.
# All others are "fixed-pattern" and intentionally repeat sequence content.
_UNIQUE_TYPES: frozenset[str] = frozenset({"RAND", "LOW0", "LOW1", "PRBS15"})


def _stable_canary_seed(base_seed: int, canary_type: str) -> int:
    """Return a process-independent seed for a canary type."""
    digest = hashlib.sha256(canary_type.encode("utf-8")).digest()
    stable_offset = int.from_bytes(digest[:8], byteorder="big") % 100_000
    return base_seed + stable_offset


# ── PRBS helpers (Fibonacci LFSR) ─────────────────────────────────────────────

def _prbs_sequence(degree: int, taps: list[int], initial_state: int, n_bits: int) -> list[int]:
    """Generate a PRBS bit sequence using a Fibonacci LFSR."""
    state = initial_state & ((1 << degree) - 1)
    if state == 0:
        state = 1
    output: list[int] = []
    for _ in range(n_bits):
        bit_out = (state >> (degree - 1)) & 1
        output.append(bit_out)
        feedback = 0
        for tap in taps:
            feedback ^= (state >> (tap - 1)) & 1
        state = ((state << 1) | feedback) & ((1 << degree) - 1)
    return output


# PRBS7: LFSR degree 7, poly x^7+x^6+1, period 127
_PRBS7_FULL  = _prbs_sequence(7,  [7, 6],   1, 127 * 4)
# PRBS15: LFSR degree 15, poly x^15+x^14+1, period 32767
_PRBS15_FULL = _prbs_sequence(15, [15, 14], 1, 32767)


def _make_sequence(canary_type: str, seed: int) -> str:
    """Return a 200-char '0'/'1' string for the given canary type and seed.

    For fixed-pattern types (ZEROS, ALT01, etc.) the seed selects a phase/offset
    within the finite set of variants for that pattern.  For random types (RAND,
    LOW0, LOW1, PRBS15) the seed drives an RNG to produce a unique sequence.
    """
    rng = random.Random(seed)

    if canary_type == "ZEROS":
        return "0" * SEQUENCE_LENGTH

    if canary_type == "ONES":
        return "1" * SEQUENCE_LENGTH

    if canary_type == "ALT01":
        # Period-2, start offset from seed
        offset = seed % 2
        return "".join(str((i + offset) % 2) for i in range(SEQUENCE_LENGTH))

    if canary_type == "ALT0011":
        # Period-4 pattern: 0,0,1,1 repeating, phase from seed
        pattern = [0, 0, 1, 1]
        offset = seed % 4
        return "".join(str(pattern[(i + offset) % 4]) for i in range(SEQUENCE_LENGTH))

    if canary_type == "RAND":
        return "".join(str(rng.randint(0, 1)) for _ in range(SEQUENCE_LENGTH))

    if canary_type == "LOW0":
        # 90 % zeros, 10 % ones placed randomly
        bits = [0] * SEQUENCE_LENGTH
        n_ones = SEQUENCE_LENGTH // 10  # 20 ones
        positions = rng.sample(range(SEQUENCE_LENGTH), n_ones)
        for pos in positions:
            bits[pos] = 1
        return "".join(str(b) for b in bits)

    if canary_type == "LOW1":
        # 90 % ones, 10 % zeros placed randomly
        bits = [1] * SEQUENCE_LENGTH
        n_zeros = SEQUENCE_LENGTH // 10  # 20 zeros
        positions = rng.sample(range(SEQUENCE_LENGTH), n_zeros)
        for pos in positions:
            bits[pos] = 0
        return "".join(str(b) for b in bits)

    if canary_type == "SAWTOOTH":
        # 3-bit counter bit 2 gives period-8 sawtooth: 0,0,0,0,1,1,1,1 repeating
        offset = seed % 8
        return "".join(str(((i + offset) // 4) % 2) for i in range(SEQUENCE_LENGTH))

    if canary_type == "PRBS7":
        # Use seed as start offset in the PRBS7 cycle (period 127)
        offset = seed % 127
        bits = [_PRBS7_FULL[(offset + i) % 127] for i in range(SEQUENCE_LENGTH)]
        return "".join(str(b) for b in bits)

    if canary_type == "PRBS15":
        # Use seed as start offset in the PRBS15 cycle (period 32767)
        offset = seed % 32767
        bits = [_PRBS15_FULL[(offset + i) % 32767] for i in range(SEQUENCE_LENGTH)]
        return "".join(str(b) for b in bits)

    if canary_type == "BLOCK":
        # Alternating blocks of 8: 00000000 11111111 ... phase from seed
        block_size = 8
        phase = seed % 2  # 0 = starts with 0-block, 1 = starts with 1-block
        bits = []
        for i in range(SEQUENCE_LENGTH):
            block_index = i // block_size
            bit = (block_index + phase) % 2
            bits.append(str(bit))
        return "".join(bits)

    if canary_type == "GRAY":
        # Gray code LSB: for counter i, gray(i) = i XOR (i>>1), emit bit 0
        # Gives pattern 0,1,1,0,0,1,1,0,... period 4
        offset = seed % 4
        return "".join(str((((i + offset) ^ ((i + offset) >> 1)) & 1))
                       for i in range(SEQUENCE_LENGTH))

    raise ValueError(f"Unknown canary type: {canary_type!r}")


def generate_canary_sequences(
    base_seed: int = 5000,
    verbose: bool = True,
    existing_sequences: set[str] | None = None,
) -> list[dict]:
    """Generate all 900 canary sequences (12 types × 75 records each).

    For fixed-pattern types (ZEROS, ONES, ALT01, etc.) the 75 records may share
    sequence content but have unique sequence_ids and varied seeds.
    For unique types (RAND, LOW0, LOW1, PRBS15) all 75 sequences are globally distinct.

    Returns a list of record dicts compatible with sequences_mvp.json.
    """
    if verbose:
        print(f"Generating {len(CANARY_TYPES) * SEQUENCES_PER_TYPE} canary sequences "
              f"({SEQUENCES_PER_TYPE}/type × {len(CANARY_TYPES)} types) ...")

    all_records: list[dict] = []
    # Track unique sequences only for types that produce them.
    # Pre-populate with any sequences from other generators to prevent cross-generator collisions.
    unique_global_seen: set[str] = set(existing_sequences) if existing_sequences else set()

    for canary_type in CANARY_TYPES:
        type_records: list[dict] = []
        # For uniqueness-enforced types, also track within-type
        unique_seen_in_type: set[str] = set()

        rng = random.Random(_stable_canary_seed(base_seed, canary_type))
        index = 0

        is_unique_type = canary_type in _UNIQUE_TYPES

        if is_unique_type:
            # Need 75 globally unique sequences; budget more attempts
            max_attempts = SEQUENCES_PER_TYPE * 500
            attempts = 0
            while len(type_records) < SEQUENCES_PER_TYPE and attempts < max_attempts:
                attempts += 1
                seed = rng.randint(0, 2**32 - 1)
                seq = _make_sequence(canary_type, seed)
                assert len(seq) == SEQUENCE_LENGTH

                if seq in unique_seen_in_type or seq in unique_global_seen:
                    continue

                unique_seen_in_type.add(seq)
                unique_global_seen.add(seq)

                type_records.append({
                    "sequence_id": f"CANARY_{canary_type}_{index:04d}",
                    "generator_type": "canary",
                    "canary_type": canary_type,
                    "seed": seed,
                    "sequence": seq,
                    "sequence_length": SEQUENCE_LENGTH,
                })
                index += 1

            if len(type_records) < SEQUENCES_PER_TYPE:
                raise RuntimeError(
                    f"Canary type {canary_type!r}: could only collect {len(type_records)} "
                    f"unique sequences after {attempts} attempts (target {SEQUENCES_PER_TYPE})."
                )
            if verbose:
                print(f"  {canary_type:<12s}: {len(type_records)} unique sequences "
                      f"(attempts={attempts})")

        else:
            # Fixed-pattern type: generate 75 records with varying seeds.
            # Sequence content may repeat (intentional); only sequence_ids are unique.
            # Use a cycle of seeds drawn from rng so each record has a different seed.
            for _ in range(SEQUENCES_PER_TYPE):
                seed = rng.randint(0, 2**32 - 1)
                seq = _make_sequence(canary_type, seed)
                assert len(seq) == SEQUENCE_LENGTH

                type_records.append({
                    "sequence_id": f"CANARY_{canary_type}_{index:04d}",
                    "generator_type": "canary",
                    "canary_type": canary_type,
                    "seed": seed,
                    "sequence": seq,
                    "sequence_length": SEQUENCE_LENGTH,
                })
                index += 1

            if verbose:
                unique_count = len({r["sequence"] for r in type_records})
                print(f"  {canary_type:<12s}: {len(type_records)} records "
                      f"({unique_count} distinct sequence(s))")

        all_records.extend(type_records)

    if verbose:
        print(f"Canary total: {len(all_records)} records")

    return all_records


def validate_canary_records(records: list[dict]) -> None:
    """Assert all invariants on the canary records."""
    assert len(records) == len(CANARY_TYPES) * SEQUENCES_PER_TYPE, (
        f"Expected {len(CANARY_TYPES) * SEQUENCES_PER_TYPE} canary records, got {len(records)}"
    )
    seen_ids: set[str] = set()

    for rec in records:
        sid = rec["sequence_id"]
        assert sid not in seen_ids, f"Duplicate sequence_id: {sid}"
        seen_ids.add(sid)

        seq = rec["sequence"]
        assert len(seq) == SEQUENCE_LENGTH, f"{sid}: length {len(seq)} != {SEQUENCE_LENGTH}"
        assert all(c in "01" for c in seq), f"{sid}: non-binary character"
        assert rec["generator_type"] == "canary"
        assert rec["canary_type"] in CANARY_TYPES, f"{sid}: unknown type {rec['canary_type']}"
        assert rec["sequence_length"] == SEQUENCE_LENGTH

    # Specific invariants for fixed-pattern canaries
    zeros_recs = [r for r in records if r["canary_type"] == "ZEROS"]
    assert len(zeros_recs) == SEQUENCES_PER_TYPE
    for r in zeros_recs:
        assert r["sequence"] == "0" * SEQUENCE_LENGTH, "ZEROS canary is not all-zeros"

    ones_recs = [r for r in records if r["canary_type"] == "ONES"]
    assert len(ones_recs) == SEQUENCES_PER_TYPE
    for r in ones_recs:
        assert r["sequence"] == "1" * SEQUENCE_LENGTH, "ONES canary is not all-ones"

    alt_recs = [r for r in records if r["canary_type"] == "ALT01"]
    assert len(alt_recs) == SEQUENCES_PER_TYPE
    for r in alt_recs:
        seq = r["sequence"]
        assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1)), (
            "ALT01 canary is not strictly alternating"
        )

    # Per-type counts
    for ct in CANARY_TYPES:
        ct_recs = [r for r in records if r["canary_type"] == ct]
        assert len(ct_recs) == SEQUENCES_PER_TYPE, (
            f"Canary type {ct}: expected {SEQUENCES_PER_TYPE}, got {len(ct_recs)}"
        )

    print("Canary validation passed.")


if __name__ == "__main__":
    records = generate_canary_sequences(verbose=True)
    validate_canary_records(records)
