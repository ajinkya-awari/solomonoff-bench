"""Tokenizer validation for binary-sequence scoring.

Builds a mapping from the model's vocabulary to accepted binary token IDs.
Accepted variants: tokens that decode (after stripping whitespace) to exactly
'0' or '1'.

Saves invalid-token mass BEFORE renormalization so callers can report it.
Renormalized scoring measures:
    P(bit | model emits a valid binary token)
NOT unconditional task compliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class BinaryTokenMap:
    """Maps 0/1 bit labels to sets of accepted token IDs for a given model."""
    model_name: str
    zero_ids: list[int] = field(default_factory=list)
    one_ids: list[int] = field(default_factory=list)
    total_vocab_size: int = 0
    # Raw string forms that decoded to 0 or 1
    zero_forms: list[str] = field(default_factory=list)
    one_forms: list[str] = field(default_factory=list)

    def is_valid(self) -> bool:
        return len(self.zero_ids) > 0 and len(self.one_ids) > 0

    def summary(self) -> dict:
        return {
            "model_name": self.model_name,
            "zero_token_ids": self.zero_ids,
            "one_token_ids": self.one_ids,
            "zero_forms": self.zero_forms,
            "one_forms": self.one_forms,
            "n_accepted_zero_tokens": len(self.zero_ids),
            "n_accepted_one_tokens": len(self.one_ids),
            "total_vocab_size": self.total_vocab_size,
        }


def build_binary_token_map(
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast",
    model_name: str,
) -> BinaryTokenMap:
    """Scan the tokenizer vocabulary and collect all token IDs whose decoded
    string (after whitespace stripping) is exactly '0' or '1'.

    Checks: "0", "1", " 0", " 1", "\\n0", "\\n1", and any single-token
    encoding of these strings produced by the tokenizer itself.
    """
    vocab = tokenizer.get_vocab()  # {str: int}
    token_map = BinaryTokenMap(
        model_name=model_name,
        total_vocab_size=len(vocab),
    )

    # Candidate surface forms
    candidate_forms = ["0", "1", " 0", " 1", "\n0", "\n1", "\t0", "\t1",
                       "0\n", "1\n", "0 ", "1 "]

    seen_zero_ids: set[int] = set()
    seen_one_ids: set[int] = set()

    # Method 1: decode every token in the vocabulary
    for token_str, token_id in vocab.items():
        stripped = token_str.strip()
        if stripped == "0" and token_id not in seen_zero_ids:
            token_map.zero_ids.append(token_id)
            token_map.zero_forms.append(repr(token_str))
            seen_zero_ids.add(token_id)
        elif stripped == "1" and token_id not in seen_one_ids:
            token_map.one_ids.append(token_id)
            token_map.one_forms.append(repr(token_str))
            seen_one_ids.add(token_id)

    # Method 2: tokenize candidate forms directly (catches BPE merge artefacts)
    for form in candidate_forms:
        ids = tokenizer.encode(form, add_special_tokens=False)
        if len(ids) == 1:
            tid = ids[0]
            decoded = tokenizer.decode([tid]).strip()
            if decoded == "0" and tid not in seen_zero_ids:
                token_map.zero_ids.append(tid)
                token_map.zero_forms.append(f"encode({form!r})→{tid}")
                seen_zero_ids.add(tid)
            elif decoded == "1" and tid not in seen_one_ids:
                token_map.one_ids.append(tid)
                token_map.one_forms.append(f"encode({form!r})→{tid}")
                seen_one_ids.add(tid)

    return token_map


def extract_binary_probs(
    log_probs_tensor,  # torch.Tensor, shape [vocab_size]
    token_map: BinaryTokenMap,
) -> dict:
    """Extract P(0) and P(1) from a log-softmax distribution.

    Returns:
        raw_p0: sum of probability mass over all accepted zero-token IDs
        raw_p1: sum of probability mass over all accepted one-token IDs
        invalid_mass: 1 - raw_p0 - raw_p1 (saved BEFORE renormalization)
        p0_renorm: P(0) renormalized so p0 + p1 = 1
        p1_renorm: P(1) renormalized so p0 + p1 = 1
    """
    import torch

    probs = torch.exp(log_probs_tensor)

    raw_p0 = sum(probs[tid].item() for tid in token_map.zero_ids)
    raw_p1 = sum(probs[tid].item() for tid in token_map.one_ids)
    invalid_mass = 1.0 - raw_p0 - raw_p1

    denom = raw_p0 + raw_p1
    if denom < 1e-12:
        # Model puts essentially no mass on valid binary tokens
        p0_renorm = 0.5
        p1_renorm = 0.5
    else:
        p0_renorm = raw_p0 / denom
        p1_renorm = raw_p1 / denom

    return {
        "raw_p0": raw_p0,
        "raw_p1": raw_p1,
        "invalid_mass": invalid_mass,
        "p0_renorm": p0_renorm,
        "p1_renorm": p1_renorm,
        "valid_binary_mass": denom,
    }
