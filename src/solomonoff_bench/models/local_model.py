"""Local HuggingFace model scorer using direct forward-pass logits.

NEVER uses model.generate(output_scores=True). That produces generation
scores, not next-token conditional probabilities.

Uses: outputs = model(**inputs); logits = outputs.logits[:, -1, :]
"""

from __future__ import annotations

from typing import Optional
import torch
import torch.nn.functional as F

from solomonoff_bench.models.base_model import BinarySequenceScorer
from solomonoff_bench.models.tokenizer_validation import (
    BinaryTokenMap,
    build_binary_token_map,
    extract_binary_probs,
)


def _build_prompt(prefix: str) -> str:
    """Format the binary prefix as a prompt for next-bit prediction."""
    formatted = ", ".join(prefix)
    return "Here is a binary sequence: " + formatted + "\nThe next symbol is:"


class LocalHFScorer(BinarySequenceScorer):
    """Scores binary sequences using a local HuggingFace causal LM.

    Forward-pass only — no sampling, no generation.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        max_seq_len: int = 512,
    ) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self._model_name = model_name_or_path
        self.max_seq_len = max_seq_len

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quant_config,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        # Set to inference mode: disables dropout, batchnorm uses running stats
        self.model.train(False)

        self.token_map: BinaryTokenMap = build_binary_token_map(
            self.tokenizer, model_name_or_path
        )
        if not self.token_map.is_valid():
            raise RuntimeError(
                "No valid binary token IDs found for model " + model_name_or_path +
                ". Cannot score binary sequences."
            )

    def model_name(self) -> str:
        return self._model_name

    @torch.no_grad()
    def score_next_bit(self, prefix: str) -> dict:
        """Forward pass to get P(next bit | prefix).

        prefix: ASCII string of '0'/'1' characters (no spaces).
        Returns dict with raw_p0, raw_p1, invalid_mass, p0_renorm, p1_renorm.
        """
        prompt = _build_prompt(prefix)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_seq_len,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Direct forward pass — NOT model.generate()
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]       # [1, vocab_size]
        log_probs = F.log_softmax(logits[0], dim=-1)  # [vocab_size]

        return extract_binary_probs(log_probs, self.token_map)

    def score_sequence_window(
        self,
        sequence: str,
        context_len: int = 100,
        predict_len: int = 100,
    ) -> list[dict]:
        """Score each symbol in the prediction window auto-regressively.

        Prefix grows from context_len symbols to context_len+predict_len-1.
        Returns list of per-symbol result dicts.
        """
        assert len(sequence) >= context_len + predict_len, (
            "Sequence too short: "
            + str(len(sequence))
            + " < "
            + str(context_len + predict_len)
        )
        results = []
        for i in range(context_len, context_len + predict_len):
            prefix = sequence[:i]
            result = self.score_next_bit(prefix)
            result["position"] = i
            result["true_bit"] = sequence[i]
            results.append(result)
        return results
