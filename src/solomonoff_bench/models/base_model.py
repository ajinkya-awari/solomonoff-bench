"""Abstract interface for binary-sequence scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BinarySequenceScorer(ABC):
    """Predict P(next_bit | prefix) for sequences of '0'/'1' characters."""

    @abstractmethod
    def score_next_bit(self, prefix: str) -> dict:
        """Return probabilities for the next bit given the prefix.

        Returns dict with at minimum:
            p0_renorm: float — P('0' | valid binary token)
            p1_renorm: float — P('1' | valid binary token)
            invalid_mass: float — probability mass outside valid binary tokens
                                  (saved before renormalization)
        """

    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    def validate_toy_sequences(self, toy_sequences: list[str]) -> dict:
        """Run the 5-prompt toy validation gate.

        For each sequence, split at the midpoint and check that
        P(0) + P(1) ≈ 1.0 (after renorm it is by construction, but
        we check that valid_binary_mass is not near zero).

        Returns diagnostics dict. Raises ValueError if gate fails.
        """
        results = []
        for seq in toy_sequences:
            mid = len(seq) // 2
            prefix = seq[:mid]
            result = self.score_next_bit(prefix)
            results.append(result)

        mean_invalid = sum(r["invalid_mass"] for r in results) / len(results)
        mean_valid_mass = sum(r["valid_binary_mass"] for r in results) / len(results)

        gate_passed = mean_valid_mass >= 0.01  # at least 1% mass on binary tokens

        if not gate_passed:
            raise ValueError(
                f"Tokenizer validation gate FAILED for {self.model_name()}. "
                f"Mean valid binary mass = {mean_valid_mass:.4f} < 0.01. "
                "The model puts essentially no probability mass on '0' or '1' tokens. "
                "Check the prompt format and tokenizer mapping."
            )

        return {
            "model_name": self.model_name(),
            "gate_passed": gate_passed,
            "n_toy_sequences": len(toy_sequences),
            "mean_invalid_mass": mean_invalid,
            "mean_valid_binary_mass": mean_valid_mass,
            "per_sequence": results,
        }
