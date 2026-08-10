"""Generate the Week 1 MVP sequence dataset.

Produces data/sequences_mvp.json:
  - 4 complexity levels (n_states = 2, 3, 4, 5)
  - 75 unique sequences per level (deduplicated on 200-symbol output)
  - Full metadata per sequence

Complexity levels map to n_states:
  Level 1 → 2 states
  Level 2 → 3 states
  Level 3 → 4 states
  Level 4 → 5 states
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

from solomonoff_bench.sequences.tm_simulator import (
    OUTPUT_LENGTH,
    MAX_TRANSITIONS,
    TMProgram,
    compute_program_bits,
    random_tm_program,
    run_tm,
)

SEQUENCES_PER_LEVEL = 75
# 2-state TMs produce only ~60 unique 200-symbol outputs (exhaustive enumeration
# confirms 4096 programs → 60 distinct outputs), which is below the 75/level target.
# We use 3–6 states to maintain the complexity gradient with sufficient diversity.
COMPLEXITY_LEVELS = {
    1: 3,  # level → n_states
    2: 4,
    3: 5,
    4: 6,
}
# Maximum TM programs to sample before giving up on a level
MAX_ATTEMPTS_PER_LEVEL = 500_000

OUTPUT_CONVENTION = "output-transducer: written symbol emitted at each transition"


def generate_level(
    complexity_level: int,
    n_states: int,
    n_sequences: int,
    base_seed: int,
    global_seen: set[str],
    verbose: bool = True,
) -> list[dict]:
    """Sample TM programs until we collect n_sequences unique accepted sequences.

    global_seen: set of sequence strings already used by earlier levels.
    New sequences are added to global_seen in place.
    """
    rng = random.Random(base_seed)
    records: list[dict] = []
    attempts = 0
    sequence_id_counter = 0

    program_bits = compute_program_bits(n_states)

    if verbose:
        print(f"  Level {complexity_level} (n_states={n_states}, "
              f"program_bits={program_bits}): ", end="", flush=True)

    while len(records) < n_sequences and attempts < MAX_ATTEMPTS_PER_LEVEL:
        seed = rng.randint(0, 2**32 - 1)
        level_rng = random.Random(seed)
        prog = random_tm_program(n_states=n_states, rng=level_rng)
        prog.seed = seed
        attempts += 1

        result = run_tm(prog)
        if not result.accepted:
            continue

        if result.sequence in global_seen:
            continue  # deduplicate globally across all levels

        global_seen.add(result.sequence)

        # Serialize transition table for JSON storage
        transition_table_json = {
            f"{state},{sym}": {
                "write": t.write,
                "direction": t.direction,
                "next_state": t.next_state,
            }
            for (state, sym), t in prog.transition_table.items()
        }

        sequence_id = f"L{complexity_level}_{sequence_id_counter:04d}"
        sequence_id_counter += 1

        records.append({
            "sequence_id": sequence_id,
            "complexity_level": complexity_level,
            "n_states": n_states,
            "program_bits": program_bits,
            "transition_table": transition_table_json,
            "seed": seed,
            "step_count": result.step_count,
            "output_convention": OUTPUT_CONVENTION,
            "sequence": result.sequence,
            "sequence_length": len(result.sequence),
        })

    if verbose:
        discard_rate = 1 - (len(records) / max(attempts, 1))
        print(f"collected {len(records)}/{n_sequences} "
              f"(attempts={attempts}, discard_rate={discard_rate:.1%})")

    if len(records) < n_sequences:
        raise RuntimeError(
            f"Could not collect {n_sequences} unique sequences for level "
            f"{complexity_level} after {attempts} attempts. Got {len(records)}."
        )

    return records


def generate_mvp_dataset(
    output_path: Path,
    base_seed: int = 2026,
    verbose: bool = True,
) -> list[dict]:
    """Generate all 300 MVP sequences and save to JSON."""
    if verbose:
        print(f"Generating {SEQUENCES_PER_LEVEL * len(COMPLEXITY_LEVELS)} sequences "
              f"({SEQUENCES_PER_LEVEL}/level × {len(COMPLEXITY_LEVELS)} levels) ...")
        print(f"Output convention: {OUTPUT_CONVENTION}")
        print(f"Sequence length: {OUTPUT_LENGTH} symbols")
        print()

    all_records: list[dict] = []
    level_seeds = {level: base_seed + level * 1000 for level in COMPLEXITY_LEVELS}
    global_seen: set[str] = set()  # shared across all levels to prevent cross-level duplicates

    t0 = time.time()
    for level, n_states in COMPLEXITY_LEVELS.items():
        level_records = generate_level(
            complexity_level=level,
            n_states=n_states,
            n_sequences=SEQUENCES_PER_LEVEL,
            base_seed=level_seeds[level],
            global_seen=global_seen,
            verbose=verbose,
        )
        all_records.extend(level_records)

    elapsed = time.time() - t0

    if verbose:
        print()
        print(f"Total sequences: {len(all_records)}")
        print(f"Generation time: {elapsed:.1f}s")

        # Per-level summary
        for level in COMPLEXITY_LEVELS:
            lvl_recs = [r for r in all_records if r["complexity_level"] == level]
            print(f"  Level {level}: {len(lvl_recs)} sequences, "
                  f"program_bits={lvl_recs[0]['program_bits']}")

    dataset = {
        "version": "mvp-week1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_seed": base_seed,
        "output_length": OUTPUT_LENGTH,
        "output_convention": OUTPUT_CONVENTION,
        "max_transitions": MAX_TRANSITIONS,
        "sequences_per_level": SEQUENCES_PER_LEVEL,
        "total_sequences": len(all_records),
        "complexity_levels": {
            str(level): {
                "n_states": n_states,
                "program_bits": compute_program_bits(n_states),
            }
            for level, n_states in COMPLEXITY_LEVELS.items()
        },
        "records": all_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    if verbose:
        print(f"\nSaved to: {output_path}")

    return all_records


def validate_dataset(records: list[dict]) -> None:
    """Assert all invariants on the generated dataset."""
    assert len(records) == SEQUENCES_PER_LEVEL * len(COMPLEXITY_LEVELS), (
        f"Expected {SEQUENCES_PER_LEVEL * len(COMPLEXITY_LEVELS)} sequences, "
        f"got {len(records)}"
    )

    seen_ids: set[str] = set()
    seen_seqs: set[str] = set()

    for rec in records:
        sid = rec["sequence_id"]
        assert sid not in seen_ids, f"Duplicate sequence_id: {sid}"
        seen_ids.add(sid)

        seq = rec["sequence"]
        assert seq not in seen_seqs, f"Duplicate sequence output for {sid}"
        seen_seqs.add(seq)

        assert len(seq) == OUTPUT_LENGTH, (
            f"{sid}: sequence length {len(seq)} != {OUTPUT_LENGTH}"
        )
        assert all(c in "01" for c in seq), (
            f"{sid}: non-binary character in sequence"
        )
        assert rec["complexity_level"] in COMPLEXITY_LEVELS
        assert rec["n_states"] == COMPLEXITY_LEVELS[rec["complexity_level"]]
        assert rec["program_bits"] == compute_program_bits(rec["n_states"])
        assert rec["output_convention"] == OUTPUT_CONVENTION

    # Check per-level counts
    for level in COMPLEXITY_LEVELS:
        level_recs = [r for r in records if r["complexity_level"] == level]
        assert len(level_recs) == SEQUENCES_PER_LEVEL, (
            f"Level {level}: expected {SEQUENCES_PER_LEVEL}, got {len(level_recs)}"
        )

    print("Validation passed: all invariants satisfied.")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[4]
    output_path = repo_root / "data" / "sequences_mvp.json"

    records = generate_mvp_dataset(output_path=output_path, verbose=True)
    validate_dataset(records)
