"""EL_gzip: gzip-normalized excess code length (Week 1 pilot metric).

EL_gzip(M, x) = H(M, x) - H_gzip_incremental(x)

where H(M, x) is the cross-entropy of model M on the prediction window and
H_gzip_incremental(x) is the incremental gzip conditional code length.

IMPORTANT: EL_gzip is NOT the Solomonoff Gap (SG). SG uses CTW as the
baseline (Week 2+). EL_gzip uses gzip as a pilot compressor heuristic only.
Never plot EL_gzip and SG on the same y-axis.
"""

from __future__ import annotations

import math
from solomonoff_bench.baselines.gzip_baseline import h_gzip_incremental


def cross_entropy_from_probs(
    true_bits: str,
    p0_renorm_list: list[float],
) -> float:
    """Compute cross-entropy in bits/symbol from per-position renorm probabilities.

    true_bits: string of '0'/'1' characters (prediction window).
    p0_renorm_list: P(0) at each position, renormalized over valid binary tokens.
    Returns mean cross-entropy in bits/symbol (always >= 0).
    """
    assert len(true_bits) == len(p0_renorm_list), (
        "Length mismatch: true_bits has "
        + str(len(true_bits))
        + " but p0_renorm_list has "
        + str(len(p0_renorm_list))
    )
    total_bits = 0.0
    for bit, p0 in zip(true_bits, p0_renorm_list):
        p_true = p0 if bit == "0" else (1.0 - p0)
        p_true = max(p_true, 1e-12)  # prevent log(0)
        total_bits += -math.log2(p_true)
    return total_bits / len(true_bits)


def compute_el_gzip(
    sequence: str,
    p0_renorm_list: list[float],
    context_len: int = 100,
    predict_len: int = 100,
) -> dict:
    """Compute EL_gzip for one sequence.

    sequence: full 200-symbol ASCII string.
    p0_renorm_list: per-position P(0) for the prediction window.
    context_len: symbols used as context.
    predict_len: symbols in prediction window.

    Returns dict with h_model, h_gzip, el_gzip, and gzip negative delta flag.
    """
    assert len(sequence) == context_len + predict_len, (
        "Sequence length "
        + str(len(sequence))
        + " != context_len + predict_len = "
        + str(context_len + predict_len)
    )

    prefix = sequence[:context_len]
    suffix = sequence[context_len:]

    h_model = cross_entropy_from_probs(suffix, p0_renorm_list)
    h_gzip_raw = h_gzip_incremental(prefix, suffix)
    gzip_negative = h_gzip_raw < 0

    el_gzip = h_model - h_gzip_raw

    return {
        "h_model_bits_per_sym": h_model,
        "h_gzip_bits_per_sym": h_gzip_raw,
        "h_gzip_is_negative": gzip_negative,
        "el_gzip": el_gzip,
        "context_len": context_len,
        "predict_len": predict_len,
    }
