"""Figure generation for Week 1 EL_gzip pilot results.

Produces:
  results/figures/fig1_mvp_el_gzip.png  — hero image for paper and README

RULES (from research.md):
  - EL_gzip and SG are NEVER plotted on the same y-axis
  - Week 1 figures never claim CTW-normalized Solomonoff Gap
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Kaggle/server
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURE_DPI = 200
FIGURE_SIZE = (7, 4.5)

MODEL_DISPLAY_NAMES = {
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3 Mini (3.8B)",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 (3B)",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama (1.1B)",
}

LEVEL_TO_LABEL = {
    1: "L1 (24 bits)",
    2: "L2 (40 bits)",
    3: "L3 (50 bits)",
    4: "L4 (60 bits)",
}

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
MARKERS = ["o", "s", "^", "D"]


def _display_name(model: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model, model.split("/")[-1])


def plot_figure1_el_gzip(
    results_csv: Path,
    output_path: Path,
    n_bootstrap: int = 5000,
) -> None:
    """Figure 1 — EL_gzip vs program_bits for all models in the CSV.

    x-axis: program_bits (complexity level proxy)
    y-axis: mean EL_gzip (bits/symbol above incremental gzip baseline)
    Error bars: 95% bootstrap confidence intervals
    One line per model.
    """
    df = pd.read_csv(results_csv)
    models = df["model"].unique().tolist()
    levels = sorted(df["complexity_level"].unique())
    x_values = [df[df["complexity_level"] == lvl]["program_bits"].iloc[0] for lvl in levels]

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    rng = np.random.default_rng(42)

    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        means, lo_errs, hi_errs = [], [], []

        for lvl in levels:
            vals = model_df[model_df["complexity_level"] == lvl]["el_gzip"].values
            mean = vals.mean()
            # Bootstrap 95% CI
            boot = rng.choice(vals, size=(n_bootstrap, len(vals)), replace=True).mean(axis=1)
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            means.append(mean)
            lo_errs.append(mean - ci_lo)
            hi_errs.append(ci_hi - mean)

        ax.errorbar(
            x_values, means,
            yerr=[lo_errs, hi_errs],
            marker=MARKERS[i % len(MARKERS)],
            color=COLORS[i % len(COLORS)],
            label=_display_name(model),
            linewidth=1.8,
            markersize=6,
            capsize=4,
            capthick=1.2,
        )

    # Random baseline reference line
    ax.axhline(0, color="gray", linestyle="--", linewidth=1,
               label="gzip baseline (reference, EL_gzip = 0)")

    ax.set_xlabel("Program bits (TM program length under fixed encoding)", fontsize=11)
    ax.set_ylabel("EL$_{gzip}$ (bits/symbol above gzip baseline)", fontsize=11)
    ax.set_title(
        "Week 1 Pilot: Excess Code Length vs TM Program Complexity\n"
        r"$\mathrm{EL_{gzip}}(M,x) = H(M,x) - H_{\mathrm{gzip\text{-}incr}}(x)$",
        fontsize=11,
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Annotation: distinguish from SG
    ax.text(
        0.98, 0.03,
        "Note: EL$_{gzip}$ ≠ SG(CTW). CTW-normalized Solomonoff\nGap is the Week 2 metric.",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
        color="gray", style="italic",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)
