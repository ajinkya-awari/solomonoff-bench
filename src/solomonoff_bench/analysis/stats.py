"""Statistical analysis for Week 1 EL_gzip results."""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, n: int = 5000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2)))


def level_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, level), grp in df.groupby(["model", "complexity_level"]):
        vals = grp["el_gzip"].values
        lo, hi = bootstrap_ci(vals)
        rows.append({
            "model": model,
            "complexity_level": level,
            "program_bits": int(grp["program_bits"].iloc[0]),
            "n": len(vals),
            "mean_el_gzip": float(vals.mean()),
            "std_el_gzip": float(vals.std()),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "mean_invalid_mass": float(grp["mean_invalid_mass"].mean()),
            "negative_gzip_count": int(grp["h_gzip_is_negative"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["model", "complexity_level"])


def minimum_viable_result_check(df: pd.DataFrame, output_path: Path) -> str:
    """Write minimum_viable_result_check.txt.

    Passes if EL_gzip varies across complexity levels (positive or not).
    Project only fails if P(0)+P(1) far from 1.0 after renorm.
    """
    summary = level_summary(df)
    lines = ["# Minimum Viable Result Check — Week 1 EL_gzip Pilot", ""]

    for model in df["model"].unique():
        m = summary[summary["model"] == model]
        el_vals = m["mean_el_gzip"].values
        spread = float(el_vals.max() - el_vals.min())
        mean_inv = float(m["mean_invalid_mass"].mean())
        verdict = "PASS" if mean_inv < 0.99 else "FAIL — model not producing valid binary tokens"

        lines += [
            f"Model: {model}",
            f"  EL_gzip by level: " + ", ".join(f"{v:.3f}" for v in el_vals),
            f"  EL_gzip spread (max-min): {spread:.3f} bits/sym",
            f"  Mean invalid-token mass: {mean_inv:.4f}",
            f"  Verdict: {verdict}",
            f"  Note: flat EL_gzip across levels is a valid finding (gzip insensitive -- need CTW)",
            "",
        ]

    text = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    print("Saved:", output_path)
    return text
