"""Build the full 2,400-sequence benchmark dataset.

Combines:
  300 TM sequences    — loaded from data/sequences_mvp.json (unchanged Week 1 artifact)
  600 CA sequences    — 8 Wolfram rules × 75 seeds
  600 LFSR sequences  — 8 LFSR degrees × 75 seeds
  900 canary sequences — 12 control types × 75 seeds
  ─────────────────────
  2400 total

Output: data/sequences_full.json

data/sequences_mvp.json is NEVER modified.

Usage:
    python scripts/build_full_dataset.py [--output PATH] [--seed N] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the src package is on the path when run as a standalone script
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from solomonoff_bench.sequences.generate_ca_sequences import (
    CA_RULES,
    SEQUENCES_PER_RULE,
    generate_ca_sequences,
    validate_ca_records,
)
from solomonoff_bench.sequences.generate_lfsr_sequences import (
    LFSR_CONFIGS,
    SEQUENCES_PER_DEGREE,
    generate_lfsr_sequences,
    validate_lfsr_records,
)
from solomonoff_bench.sequences.generate_canary_sequences import (
    CANARY_TYPES,
    SEQUENCES_PER_TYPE,
    generate_canary_sequences,
    validate_canary_records,
)

# Expected totals
TM_COUNT = 300
CA_COUNT = len(CA_RULES) * SEQUENCES_PER_RULE          # 8 × 75 = 600
LFSR_COUNT = len(LFSR_CONFIGS) * SEQUENCES_PER_DEGREE  # 8 × 75 = 600
CANARY_COUNT = len(CANARY_TYPES) * SEQUENCES_PER_TYPE  # 12 × 75 = 900
TOTAL_COUNT = TM_COUNT + CA_COUNT + LFSR_COUNT + CANARY_COUNT  # 2400


def load_tm_records(mvp_path: Path) -> list[dict]:
    """Load TM records from sequences_mvp.json, preserving all original fields."""
    if not mvp_path.exists():
        raise FileNotFoundError(f"sequences_mvp.json not found at {mvp_path}")

    with open(mvp_path, encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = data["records"]
    print(f"Loaded {len(records)} TM records from {mvp_path}")

    if len(records) != TM_COUNT:
        raise ValueError(
            f"Expected {TM_COUNT} TM records in sequences_mvp.json, found {len(records)}"
        )

    # Patch in generator_type for uniformity (TM records don't have it in week 1)
    for rec in records:
        rec.setdefault("generator_type", "tm")

    return records


def validate_full_dataset(records: list[dict]) -> None:
    """Assert all global invariants on the combined 2400-record dataset."""
    print(f"\nValidating full dataset ({len(records)} records) ...")

    assert len(records) == TOTAL_COUNT, (
        f"Expected {TOTAL_COUNT} total records, got {len(records)}"
    )

    # Canary fixed-pattern types intentionally repeat sequence content (e.g. 75 ZEROS records
    # all have the same "000...0" string).  Only sequence_ids must be globally unique.
    # For non-canary and unique-canary types, sequence content should be unique too.
    from solomonoff_bench.sequences.generate_canary_sequences import _UNIQUE_TYPES as _CANARY_UNIQUE

    seen_ids: set[str] = set()
    seen_non_canary_seqs: set[str] = set()

    for rec in records:
        sid = rec["sequence_id"]
        assert sid not in seen_ids, f"Duplicate sequence_id: {sid}"
        seen_ids.add(sid)

        seq = rec["sequence"]
        assert len(seq) == 200, f"{sid}: sequence length {len(seq)} != 200"
        assert all(c in "01" for c in seq), f"{sid}: non-binary character"
        assert rec.get("sequence_length") == 200, f"{sid}: sequence_length field != 200"

        # Enforce unique sequence content for TM, CA, LFSR and unique canary types
        gt = rec.get("generator_type", "unknown")
        ct = rec.get("canary_type", "")
        is_fixed_canary = (gt == "canary" and ct not in _CANARY_UNIQUE)
        if not is_fixed_canary:
            assert seq not in seen_non_canary_seqs, (
                f"Duplicate sequence content for non-fixed record {sid}"
            )
            seen_non_canary_seqs.add(seq)

    # Generator-type counts
    by_type: dict[str, int] = {}
    for rec in records:
        gt = rec.get("generator_type", "unknown")
        by_type[gt] = by_type.get(gt, 0) + 1

    assert by_type.get("tm", 0) == TM_COUNT, (
        f"TM count mismatch: expected {TM_COUNT}, got {by_type.get('tm', 0)}"
    )
    assert by_type.get("ca", 0) == CA_COUNT, (
        f"CA count mismatch: expected {CA_COUNT}, got {by_type.get('ca', 0)}"
    )
    assert by_type.get("lfsr", 0) == LFSR_COUNT, (
        f"LFSR count mismatch: expected {LFSR_COUNT}, got {by_type.get('lfsr', 0)}"
    )
    assert by_type.get("canary", 0) == CANARY_COUNT, (
        f"Canary count mismatch: expected {CANARY_COUNT}, got {by_type.get('canary', 0)}"
    )

    print("Full dataset validation passed.")
    print(f"  TM records    : {by_type.get('tm', 0)}")
    print(f"  CA records    : {by_type.get('ca', 0)}")
    print(f"  LFSR records  : {by_type.get('lfsr', 0)}")
    print(f"  Canary records: {by_type.get('canary', 0)}")
    print(f"  Total         : {len(records)}")


def build_full_dataset(
    mvp_path: Path,
    output_path: Path,
    ca_seed: int = 3000,
    lfsr_seed: int = 4000,
    canary_seed: int = 5000,
    verbose: bool = True,
) -> list[dict]:
    """Build and save the full 2,400-sequence dataset."""
    t0 = time.time()

    # 1. Load TM records (unchanged Week 1 artifact)
    tm_records = load_tm_records(mvp_path)

    # Build a cumulative seen-set shared across all generators to prevent cross-generator
    # sequence collisions.  Each generator extends the set with its own sequences.
    shared_seen: set[str] = {r["sequence"] for r in tm_records}

    # 2. Generate CA sequences (deduplicate against TM)
    if verbose:
        print()
    ca_records = generate_ca_sequences(
        base_seed=ca_seed, verbose=verbose, existing_sequences=shared_seen
    )
    validate_ca_records(ca_records)
    shared_seen.update(r["sequence"] for r in ca_records)

    # 3. Generate LFSR sequences (deduplicate against TM + CA)
    if verbose:
        print()
    lfsr_records = generate_lfsr_sequences(
        base_seed=lfsr_seed, verbose=verbose, existing_sequences=shared_seen
    )
    validate_lfsr_records(lfsr_records)
    shared_seen.update(r["sequence"] for r in lfsr_records)

    # 4. Generate canary sequences (unique types deduplicate against TM + CA + LFSR)
    if verbose:
        print()
    canary_records = generate_canary_sequences(
        base_seed=canary_seed, verbose=verbose, existing_sequences=shared_seen
    )
    validate_canary_records(canary_records)

    # 5. Combine all records
    all_records = tm_records + ca_records + lfsr_records + canary_records

    # 6. Global validation
    validate_full_dataset(all_records)

    # 7. Build the output JSON document
    dataset = {
        "version": "full-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_sequences": len(all_records),
        "sequence_length": 200,
        "sources": {
            "tm": {
                "count": TM_COUNT,
                "source_file": "data/sequences_mvp.json",
                "description": "Week 1 TM-generated sequences (unchanged)",
            },
            "ca": {
                "count": CA_COUNT,
                "rules": CA_RULES,
                "sequences_per_rule": SEQUENCES_PER_RULE,
                "base_seed": ca_seed,
            },
            "lfsr": {
                "count": LFSR_COUNT,
                "degrees": [d for d, _ in LFSR_CONFIGS],
                "sequences_per_degree": SEQUENCES_PER_DEGREE,
                "base_seed": lfsr_seed,
            },
            "canary": {
                "count": CANARY_COUNT,
                "types": CANARY_TYPES,
                "sequences_per_type": SEQUENCES_PER_TYPE,
                "base_seed": canary_seed,
            },
        },
        "records": all_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    elapsed = time.time() - t0
    if verbose:
        print(f"\nSaved {len(all_records)} records to: {output_path}")
        print(f"Generation time: {elapsed:.1f}s")

    return all_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_REPO_ROOT / "data" / "sequences_full.json",
        help="Output path for the full dataset (default: data/sequences_full.json)",
    )
    parser.add_argument(
        "--mvp",
        type=Path,
        default=_REPO_ROOT / "data" / "sequences_mvp.json",
        help="Path to the existing MVP TM dataset (default: data/sequences_mvp.json)",
    )
    parser.add_argument("--ca-seed",     type=int, default=3000)
    parser.add_argument("--lfsr-seed",   type=int, default=4000)
    parser.add_argument("--canary-seed", type=int, default=5000)
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    build_full_dataset(
        mvp_path=args.mvp,
        output_path=args.output,
        ca_seed=args.ca_seed,
        lfsr_seed=args.lfsr_seed,
        canary_seed=args.canary_seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
