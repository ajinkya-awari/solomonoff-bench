"""Solomonoff Gap (SG) metric — Week 2+ research metric.

SG(M, x) = H(M, x) - H_CTW(x)   [bits per symbol]

where:
    H(M, x)   = LLM cross-entropy on the prediction window
    H_CTW(x)  = CTW cross-entropy on the same window given the same context

A negative SG means the model outperformed CTW on this sequence.
This can indicate memorization (check canary sequences), genuine
pattern learning for structures CTW handles poorly at short context,
or CTW under-convergence at n=100 context symbols.

IMPORTANT: SG is NOT EL_gzip.  EL_gzip uses incremental gzip as the
baseline.  SG uses CTW.  They are different quantities and must never be
plotted on the same axis.

IMPORTANT: CTW is NOT the true Solomonoff bound.  It is the best known
practical approximation for binary sequences.  SG is therefore a lower
bound on the true Solomonoff gap.

Design note: we report SG at D=8 as primary and also log D=4 and D=12.
The reported SG uses the minimum across depths (gap against the *best*
CTW, not an arbitrary fixed-depth one):
    SG_min(M, x) = H(M, x) - max(H_CTW_D4, H_CTW_D8, H_CTW_D12)
Taking max of H_CTW (== min gap) is the conservative choice: it gives
CTW the best possible baseline, so any positive SG cannot be explained
away by depth selection.
"""

from __future__ import annotations

from solomonoff_bench.baselines.ctw import CTW
from solomonoff_bench.metrics.excess_loss import cross_entropy_from_probs

_DEFAULT_DEPTHS = (4, 8, 12)


def compute_sg(
    sequence: str,
    p0_renorm_list: list[float],
    *,
    context_len: int = 100,
    predict_len: int = 100,
    depths: tuple[int, ...] = _DEFAULT_DEPTHS,
) -> dict:
    """Compute Solomonoff Gap for one sequence.

    Parameters
    ----------
    sequence : str
        Full 200-symbol ASCII '0'/'1' string.
    p0_renorm_list : list[float]
        Per-position renormalized P(next=0) from the LLM, length = predict_len.
    context_len : int
        Number of context symbols (default 100).
    predict_len : int
        Number of prediction symbols (default 100).
    depths : tuple[int, ...]
        CTW tree depths to evaluate.  SG is reported at each depth and at
        the minimum-gap depth (most conservative).

    Returns
    -------
    dict with:
        h_model_bits_per_sym   : float — LLM cross-entropy
        h_ctw_by_depth         : dict[int, float] — CTW entropy at each depth
        h_ctw_primary          : float — H_CTW at D=8 (or first depth)
        h_ctw_best             : float — max H_CTW across depths (min gap)
        sg_by_depth            : dict[int, float] — SG at each depth
        sg_primary             : float — SG at D=8 (primary reported metric)
        sg_min                 : float — SG using best CTW (conservative)
        context_len            : int
        predict_len            : int
    """
    assert len(sequence) == context_len + predict_len, (
        f"sequence length {len(sequence)} != "
        f"context_len {context_len} + predict_len {predict_len}"
    )
    assert all(c in "01" for c in sequence), "sequence must be ASCII '0'/'1' string"
    assert len(p0_renorm_list) == predict_len, (
        f"p0_renorm_list length {len(p0_renorm_list)} != predict_len {predict_len}"
    )

    suffix = sequence[context_len:]
    h_model = cross_entropy_from_probs(suffix, p0_renorm_list)

    ctw_runner = CTW(depth=max(depths))  # reuse for score_sequence_str
    h_ctw_by_depth: dict[int, float] = {}
    for d in depths:
        result = ctw_runner.score_sequence_str(
            sequence,
            context_len=context_len,
            predict_len=predict_len,
            depth=d,
        )
        h_ctw_by_depth[d] = result["h_ctw_bits_per_sym"]

    h_ctw_best = max(h_ctw_by_depth.values())   # max H_CTW = min gap = most conservative
    primary_depth = 8 if 8 in depths else depths[0]
    h_ctw_primary = h_ctw_by_depth[primary_depth]

    sg_by_depth = {d: h_model - h for d, h in h_ctw_by_depth.items()}
    sg_primary = h_model - h_ctw_primary
    sg_min = h_model - h_ctw_best

    return {
        "h_model_bits_per_sym": h_model,
        "h_ctw_by_depth": h_ctw_by_depth,
        "h_ctw_primary": h_ctw_primary,
        "h_ctw_best": h_ctw_best,
        "sg_by_depth": sg_by_depth,
        "sg_primary": sg_primary,
        "sg_min": sg_min,
        "context_len": context_len,
        "predict_len": predict_len,
    }
