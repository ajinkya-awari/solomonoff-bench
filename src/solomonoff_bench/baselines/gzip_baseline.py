"""Incremental gzip baseline for Week 1 EL_gzip pilot.

The sequence is stored as an ASCII string of '0'/'1' characters. Each
character is encoded as one byte. len(gzip(str.encode('ascii'))) is the
compressed byte count; multiply by 8 for bits.

IMPORTANT: use the incremental formula, not the whole-string ratio.
Whole-string gzip ratio is dominated by header overhead for short windows.

Negative deltas are LOGGED and REPORTED — never silently clamped.
"""

from __future__ import annotations

import gzip


def h_gzip_incremental(prefix: str, suffix: str) -> float:
    """Conditional code-length estimate: bits per symbol for suffix given prefix.

    prefix and suffix must be ASCII strings of '0'/'1' characters.
    Returns a float that can be negative (DEFLATE block/Huffman decisions
    can make compressed(prefix+suffix) < compressed(prefix) for short inputs).
    Callers must not silently clamp or discard negative values.
    """
    assert all(c in "01" for c in prefix), "prefix must be '0'/'1' ASCII string"
    assert all(c in "01" for c in suffix), "suffix must be '0'/'1' ASCII string"
    assert len(suffix) > 0, "suffix must be non-empty"

    p_bytes = prefix.encode("ascii")
    f_bytes = (prefix + suffix).encode("ascii")
    compressed_prefix = len(gzip.compress(p_bytes, compresslevel=9))
    compressed_full = len(gzip.compress(f_bytes, compresslevel=9))
    return 8 * (compressed_full - compressed_prefix) / len(suffix)


class GzipBaseline:
    """Stateless gzip baseline scorer.

    Tracks all negative-delta occurrences for reporting.
    """

    def __init__(self) -> None:
        self.negative_delta_count: int = 0
        self.total_count: int = 0
        self._negative_deltas: list[float] = []

    def score(self, prefix: str, suffix: str) -> float:
        """Return incremental gzip code length (bits/symbol).

        Negative values are recorded in diagnostics — not clamped.
        """
        value = h_gzip_incremental(prefix, suffix)
        self.total_count += 1
        if value < 0:
            self.negative_delta_count += 1
            self._negative_deltas.append(value)
        return value

    @property
    def negative_delta_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.negative_delta_count / self.total_count

    def diagnostics(self) -> dict:
        return {
            "total_calls": self.total_count,
            "negative_delta_count": self.negative_delta_count,
            "negative_delta_rate": self.negative_delta_rate,
            "negative_delta_values": self._negative_deltas,
            "note": (
                "Negative deltas occur when gzip DEFLATE decisions make "
                "len(compress(prefix+suffix)) < len(compress(prefix)). "
                "They are logged here and reported in the paper, not clamped."
            ),
        }

    def reset(self) -> None:
        self.negative_delta_count = 0
        self.total_count = 0
        self._negative_deltas = []
