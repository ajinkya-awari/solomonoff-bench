"""Context Tree Weighting (CTW) baseline for the Solomonoff Gap metric.

Implements the algorithm from Willems, Shtarkov & Tjalkens (1995) for
binary sequences with Krichevsky-Trofimov (KT) estimators at each node.

CTW is the best known practical approximation to Solomonoff induction for
binary sequences. It is parameter-free (given depth D), converges to the
optimal code length for tree-structured sources, and achieves redundancy
O(log n / n) per symbol.

IMPORTANT: CTW is NOT the true Solomonoff bound. It is the best known
practical approximation to it. SG = H(M, x) - H_CTW(x) is therefore a
lower bound on the true Solomonoff gap.

Tree convention
---------------
Context path for predicting x_{t+1} given history x_1..x_t:
    path = [x_t, x_{t-1}, ..., x_{t-D+1}]   (most-recent first, length min(t, D))
Root is depth 0; leaf is depth D.
At depth d, descend into child path[d].
Counts at each node accumulate over all past observations that passed
through that node.

KT estimator (Krichevsky-Trofimov)
------------------------------------
P_kt(next=0 | a zeros, b ones seen) = (a + 0.5) / (a + b + 1)
P_kt(next=1 | a zeros, b ones seen) = (b + 0.5) / (a + b + 1)

Weighted mixture (the CTW formula)
------------------------------------
At a leaf (depth == D or no more context):
    P_w = P_kt

At an internal node:
    P_w(bit) = 0.5 * P_kt(bit) + 0.5 * P_w_child(bit)

where P_w_child is the weighted probability from the subtree of the child
corresponding to the current context bit.
"""

from __future__ import annotations

import math
from typing import Optional


class _Node:
    """A single node in the context tree."""

    __slots__ = ("a", "b", "children")

    def __init__(self) -> None:
        self.a: float = 0.0   # count of 0s observed through this node
        self.b: float = 0.0   # count of 1s observed through this node
        self.children: dict[int, "_Node"] = {}

    def kt_prob(self, bit: int) -> float:
        """KT sequential probability estimate for the next bit."""
        if bit == 0:
            return (self.a + 0.5) / (self.a + self.b + 1.0)
        return (self.b + 0.5) / (self.a + self.b + 1.0)


class CTW:
    """Context Tree Weighting for binary sequences.

    Parameters
    ----------
    depth : int
        Maximum context depth D.  Recommended values: 4, 8, 12.
        Deeper trees are more accurate but use more memory.
        CTW with depth D requires ~2^D symbols to converge.
    """

    def __init__(self, depth: int = 8) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth
        self._root = _Node()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, history: list[int]) -> tuple[float, float]:
        """Return (P(next=0), P(next=1)) from the current CTW distribution.

        Does NOT update internal counts.  Call update() after predicting.
        """
        path = self._context_path(history)
        p0 = self._predict_node(self._root, path, 0, 0)
        p1 = self._predict_node(self._root, path, 0, 1)
        # Should sum to 1 by construction; normalise for floating-point safety.
        total = p0 + p1
        return p0 / total, p1 / total

    def update(self, history: list[int], bit: int) -> float:
        """Observe `bit` after `history`.  Returns P(bit | history).

        Updates internal counts along the full context path so subsequent
        calls benefit from this observation.
        """
        path = self._context_path(history)
        return self._update_node(self._root, path, 0, bit)

    def score_sequence(
        self,
        context: list[int],
        prediction: list[int],
        *,
        depth: Optional[int] = None,
    ) -> dict:
        """Score a prediction window given a context using a FRESH CTW tree.

        A new CTW instance is created per call so sequences are independent.

        Parameters
        ----------
        context : list[int]
            Past symbols (0/1) used to warm up the CTW tree.
        prediction : list[int]
            Symbols to score (0/1).
        depth : int, optional
            Override tree depth for this call; defaults to self.depth.

        Returns
        -------
        dict with:
            h_ctw_bits_per_sym : float  — mean cross-entropy (bits/symbol)
            depth               : int   — tree depth used
            n_context           : int
            n_predicted         : int
        """
        d = depth if depth is not None else self.depth
        ctw = CTW(depth=d)

        history: list[int] = []
        for bit in context:
            ctw.update(history, bit)
            history.append(bit)

        neg_log_probs: list[float] = []
        for bit in prediction:
            p = ctw.update(history, bit)
            neg_log_probs.append(-math.log2(max(p, 1e-12)))
            history.append(bit)

        return {
            "h_ctw_bits_per_sym": sum(neg_log_probs) / len(neg_log_probs),
            "depth": d,
            "n_context": len(context),
            "n_predicted": len(prediction),
        }

    def score_sequence_str(
        self,
        sequence: str,
        *,
        context_len: int = 100,
        predict_len: int = 100,
        depth: Optional[int] = None,
    ) -> dict:
        """Convenience wrapper accepting the ASCII '0'/'1' string format.

        sequence : 200-character ASCII string of '0' and '1'.
        Returns the same dict as score_sequence() plus context_len / predict_len.
        """
        assert len(sequence) == context_len + predict_len, (
            f"sequence length {len(sequence)} != "
            f"context_len {context_len} + predict_len {predict_len}"
        )
        assert all(c in "01" for c in sequence), "sequence must be ASCII '0'/'1' string"

        context = [int(c) for c in sequence[:context_len]]
        prediction = [int(c) for c in sequence[context_len:]]
        result = self.score_sequence(context, prediction, depth=depth)
        result["context_len"] = context_len
        result["predict_len"] = predict_len
        return result

    def reset(self) -> None:
        """Reset the tree to its initial (empty) state."""
        self._root = _Node()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _context_path(self, history: list[int]) -> list[int]:
        """Most-recent-first context of length min(len(history), depth)."""
        n = min(len(history), self.depth)
        return list(reversed(history[-n:])) if n > 0 else []

    def _predict_node(
        self, node: _Node, path: list[int], depth: int, bit: int
    ) -> float:
        """Weighted predictive probability P_w(bit) using current counts (no update)."""
        p_kt = node.kt_prob(bit)

        if depth >= len(path) or depth >= self.depth:
            return p_kt  # leaf: pure KT

        ctx_bit = path[depth]
        if ctx_bit not in node.children:
            # Unseen child — empty subtree has P_w = 0.5 everywhere.
            return 0.5 * p_kt + 0.5 * 0.5

        p_child = self._predict_node(node.children[ctx_bit], path, depth + 1, bit)
        return 0.5 * p_kt + 0.5 * p_child

    def _update_node(
        self, node: _Node, path: list[int], depth: int, bit: int
    ) -> float:
        """Update counts and return P_w(bit) before the update (sequential prob)."""
        p_kt = node.kt_prob(bit)  # read BEFORE updating counts

        if depth >= len(path) or depth >= self.depth:
            # Leaf: update count, return KT prob.
            if bit == 0:
                node.a += 1
            else:
                node.b += 1
            return p_kt

        ctx_bit = path[depth]
        if ctx_bit not in node.children:
            node.children[ctx_bit] = _Node()

        p_child = self._update_node(node.children[ctx_bit], path, depth + 1, bit)
        p_w = 0.5 * p_kt + 0.5 * p_child

        # Update count at this node after recursing.
        if bit == 0:
            node.a += 1
        else:
            node.b += 1

        return p_w
