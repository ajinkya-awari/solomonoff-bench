"""Compute Solomonoff Gap (SG) at CTW depths D=4, 8, 12.

Loads:
  data/sequences_mvp.json          — 300 sequences (records[].sequence, 200-char ASCII)
  results/el_gzip_results_combined.csv — h_model_bits_per_sym per (sequence_id, model)

Computes:
  CTW at D=4, 8, 12 for each sequence (independent of model)
  SG = h_model_bits_per_sym - h_ctw_bits_per_sym  at each depth
  sg_primary  = SG at D=8
  sg_min      = h_model - max(h_ctw_D4, h_ctw_D8, h_ctw_D12)  [conservative]

Writes:
  results/sg_ctw_results.csv
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

# --- Path setup ---------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from solomonoff_bench.baselines.ctw import CTW  # noqa: E402

# --- Config -------------------------------------------------------------
DEPTHS = (4, 8, 12)
CONTEXT_LEN = 100
PREDICT_LEN = 100

DATA_PATH = REPO_ROOT / "data" / "sequences_mvp.json"
RESULTS_PATH = REPO_ROOT / "results" / "el_gzip_results_combined.csv"
OUTPUT_PATH = REPO_ROOT / "results" / "sg_ctw_results.csv"
FIGURE_PATH = REPO_ROOT / "results" / "figures" / "fig1_v2_sg_ctw.png"


# --- Load data ----------------------------------------------------------

def load_sequences(path: Path) -> dict[str, str]:
    """Return {sequence_id: sequence_string} for all 300 records."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = data["records"]
    mapping = {r["sequence_id"]: r["sequence"] for r in records}
    assert all(len(s) == CONTEXT_LEN + PREDICT_LEN for s in mapping.values()), \
        "Not all sequences are 200 symbols"
    return mapping


def load_model_entropies(path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: sequence_id, model, h_model_bits_per_sym, program_bits, complexity_level."""
    df = pd.read_csv(path)
    needed = {"sequence_id", "model", "h_model_bits_per_sym", "program_bits", "complexity_level"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"el_gzip CSV missing columns: {missing}")
    return df[list(needed)].copy()


# --- CTW scoring --------------------------------------------------------

def ctw_score_sequence(sequence: str, depths: tuple[int, ...]) -> dict[int, float]:
    """Score one sequence at multiple CTW depths. Returns {depth: h_ctw_bits_per_sym}."""
    results: dict[int, float] = {}
    for d in depths:
        ctw = CTW(depth=d)
        out = ctw.score_sequence_str(
            sequence,
            context_len=CONTEXT_LEN,
            predict_len=PREDICT_LEN,
        )
        results[d] = out["h_ctw_bits_per_sym"]
    return results


def compute_all_ctw(
    sequences: dict[str, str],
    depths: tuple[int, ...],
) -> pd.DataFrame:
    """Compute CTW entropy at all depths for every sequence. Returns tidy DataFrame."""
    rows = []
    total = len(sequences)
    t0 = time.time()
    for i, (seq_id, seq) in enumerate(sequences.items()):
        h_ctw_by_depth = ctw_score_sequence(seq, depths)
        row: dict = {"sequence_id": seq_id}
        for d, h in h_ctw_by_depth.items():
            row[f"h_ctw_D{d}"] = h
        row["h_ctw_best"] = max(h_ctw_by_depth.values())  # max H_CTW = min gap
        rows.append(row)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  CTW progress: {i+1}/{total} sequences — {elapsed:.1f}s elapsed")
    return pd.DataFrame(rows)


# --- SG computation -----------------------------------------------------

def compute_sg_df(model_df: pd.DataFrame, ctw_df: pd.DataFrame) -> pd.DataFrame:
    """Merge model entropies with CTW scores and compute SG columns."""
    merged = model_df.merge(ctw_df, on="sequence_id", how="inner")
    for d in DEPTHS:
        col = f"h_ctw_D{d}"
        merged[f"sg_D{d}"] = merged["h_model_bits_per_sym"] - merged[col]

    # Primary metric (D=8) and conservative minimum-gap metric
    merged["sg_primary"] = merged["sg_D8"]
    merged["sg_min"] = merged["h_model_bits_per_sym"] - merged["h_ctw_best"]

    return merged


# --- Figure 2 -----------------------------------------------------------

def plot_sg_figure(df: pd.DataFrame, output_path: Path) -> None:
    """Figure 1 v2 — SG_primary vs program_bits per model with bootstrap CIs."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]
    MARKERS = ["o", "s", "^", "D"]

    MODEL_DISPLAY_NAMES = {
        "Qwen/Qwen2.5-3B": "Qwen2.5-3B (base)",
        "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B (base)",
        "microsoft/Phi-3-mini-4k-instruct": "Phi-3 Mini (3.8B)",
        "meta-llama/Llama-3.2-3B-Instruct": "Llama 3.2 (3B)",
    }

    def display_name(m: str) -> str:
        return MODEL_DISPLAY_NAMES.get(m, m.split("/")[-1])

    models = df["model"].unique().tolist()
    levels = sorted(df["complexity_level"].unique())
    x_values = [df[df["complexity_level"] == lvl]["program_bits"].iloc[0] for lvl in levels]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
    rng = np.random.default_rng(42)
    n_bootstrap = 5000

    for i, model in enumerate(models):
        mdf = df[df["model"] == model]
        means, lo_errs, hi_errs = [], [], []
        for lvl in levels:
            vals = mdf[mdf["complexity_level"] == lvl]["sg_primary"].values
            mean = vals.mean()
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
            label=display_name(model),
            linewidth=1.8,
            markersize=6,
            capsize=4,
            capthick=1.2,
        )

    # Zero reference — SG=0 means model matches CTW
    ax.axhline(0, color="gray", linestyle="--", linewidth=1,
               label="CTW baseline (SG = 0)")

    ax.set_xlabel("Program bits (TM program length under fixed encoding)", fontsize=11)
    ax.set_ylabel("SG$_{\\mathrm{primary}}$ (bits/symbol above CTW, D=8)", fontsize=11)
    ax.set_title(
        "Week 2: Solomonoff Gap vs TM Program Complexity\n"
        r"$\mathrm{SG}(M,x) = H(M,x) - H_{\mathrm{CTW\,D=8}}(x)$",
        fontsize=11,
    )
    ax.set_xticks(x_values)
    ax.set_xticklabels([str(x) for x in x_values])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    ax.text(
        0.98, 0.03,
        "Note: SG(CTW) ≠ EL$_{gzip}$. EL$_{gzip}$ (Week 1) used gzip;\nSG uses CTW D=8 as baseline.",
        transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
        color="gray", style="italic",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# --- Main ---------------------------------------------------------------

def main() -> None:
    print("Loading sequences from", DATA_PATH)
    sequences = load_sequences(DATA_PATH)
    print(f"  {len(sequences)} sequences loaded")

    print("Loading model entropies from", RESULTS_PATH)
    model_df = load_model_entropies(RESULTS_PATH)
    models = model_df["model"].unique().tolist()
    print(f"  {len(model_df)} rows — models: {models}")

    print(f"\nComputing CTW at depths {DEPTHS} for {len(sequences)} sequences...")
    ctw_df = compute_all_ctw(sequences, DEPTHS)
    print(f"  CTW done — {len(ctw_df)} rows")

    print("\nComputing Solomonoff Gap...")
    sg_df = compute_sg_df(model_df, ctw_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sg_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(sg_df)} rows to {OUTPUT_PATH}")

    # Summary stats
    print("\n--- SG summary (mean ± std) by model and complexity level ---")
    summary = sg_df.groupby(["model", "complexity_level"])["sg_primary"].agg(["mean", "std", "count"])
    print(summary.to_string())

    negative_sg = (sg_df["sg_primary"] < 0).sum()
    print(f"\nNegative SG rows (model outperforms CTW D=8): {negative_sg}/{len(sg_df)}")

    print("\nGenerating Figure 1 v2 (SG vs program_bits)...")
    plot_sg_figure(sg_df, FIGURE_PATH)

    print("\nDone.")


if __name__ == "__main__":
    main()
