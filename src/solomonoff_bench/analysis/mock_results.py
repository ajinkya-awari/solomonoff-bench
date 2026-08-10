"""Synthetic EL_gzip results generator for pipeline smoke testing.

Generates a realistic mock el_gzip_results.csv so that plots.py and
stats.py can be tested without running Kaggle GPU inference.

The mock data simulates the expected finding: EL_gzip rises with program_bits
(models lose more to the gzip baseline as complexity increases), with realistic
noise and some negative gzip deltas.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

# Expected EL_gzip means per level per model (rough estimate based on prior work)
# Higher complexity_level → higher EL_gzip
MOCK_PROFILES = {
    "microsoft/Phi-3-mini-4k-instruct": {1: 0.12, 2: 0.28, 3: 0.45, 4: 0.61},
    "meta-llama/Llama-3.2-3B-Instruct":  {1: 0.09, 2: 0.23, 3: 0.40, 4: 0.55},
}

LEVEL_PROGRAM_BITS = {1: 24, 2: 40, 3: 50, 4: 60}
LEVEL_N_STATES     = {1: 3,  2: 4,  3: 5,  4: 6}
SEQUENCES_PER_LEVEL = 75


def generate_mock_results(
    output_path: Path,
    seed: int = 42,
    noise_std: float = 0.18,
    neg_gzip_rate: float = 0.03,
    invalid_mass_mean: float = 0.12,
) -> Path:
    """Write a realistic synthetic el_gzip_results.csv."""
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sequence_id", "complexity_level", "n_states", "program_bits",
        "model", "context_len", "predict_len",
        "h_model_bits_per_sym", "h_gzip_bits_per_sym", "h_gzip_is_negative",
        "el_gzip", "mean_invalid_mass", "mean_valid_binary_mass",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model, profile in MOCK_PROFILES.items():
            for level in range(1, 5):
                pb = LEVEL_PROGRAM_BITS[level]
                ns = LEVEL_N_STATES[level]
                el_mean = profile[level]

                for seq_idx in range(SEQUENCES_PER_LEVEL):
                    sid = "L" + str(level) + "_" + str(seq_idx).zfill(4)
                    el = el_mean + rng.gauss(0, noise_std)
                    h_gzip = 0.82 + rng.gauss(0, 0.15)
                    is_neg = rng.random() < neg_gzip_rate
                    if is_neg:
                        h_gzip = -abs(rng.gauss(0.05, 0.03))
                    h_model = el + h_gzip
                    inv_mass = max(0.0, min(0.99, invalid_mass_mean + rng.gauss(0, 0.04)))

                    writer.writerow({
                        "sequence_id": sid,
                        "complexity_level": level,
                        "n_states": ns,
                        "program_bits": pb,
                        "model": model,
                        "context_len": 100,
                        "predict_len": 100,
                        "h_model_bits_per_sym": round(h_model, 6),
                        "h_gzip_bits_per_sym": round(h_gzip, 6),
                        "h_gzip_is_negative": is_neg,
                        "el_gzip": round(el, 6),
                        "mean_invalid_mass": round(inv_mass, 6),
                        "mean_valid_binary_mass": round(1.0 - inv_mass, 6),
                    })

    print("Mock results written to:", output_path)
    return output_path
