# CLAUDE.md — solomonoff-bench

## What This Project Is

Empirical benchmark measuring how many extra bits per symbol LLMs waste vs the theoretically optimal predictor (Solomonoff induction), as a function of Kolmogorov complexity. Week 1 metric: EL_gzip (gzip-normalized excess code length).

## Stack

Python 3.10 · PyTorch ≥2.1 · HuggingFace Transformers ≥4.40 · gzip (stdlib) · ruff · pytest

Primary models: `meta-llama/Llama-3.2-3B-Instruct` + `microsoft/Phi-3-mini-4k-instruct` on Kaggle T4

## Commands

```
python -m pytest tests/ -v                           # full test suite
python src/tm_simulator.py --validate                # TM unit tests
python src/tokenizer_validator.py --model <name>     # tokenizer gate
python src/scorer.py --validate-only --model <name>  # 5-prompt toy gate
python src/scorer.py --model <name> --checkpoint-every 50
python src/generate_sequences.py                     # build dataset
ruff check src/ --fix                                # lint
```

## Architecture

```
src/
  tm_simulator.py       — binary TM, output-transducer convention, 200-symbol sequences
  generate_sequences.py — 300 sequences (75 × 4 complexity levels), deduplication
  tokenizer_validator.py — token ID collection, invalid-mass logging, renormalization
  scorer.py             — EL_gzip computation, incremental gzip, logit extraction
data/
  sequences_mvp.json    — gitignored, never commit
results/
  raw_predictions.csv   — per-sequence model log-probs
  el_gzip_results.csv   — final EL_gzip scores
  figures/              — paper figures (PDF + PNG)
tests/
  test_tm_simulator.py  — output-transducer unit tests
notebooks/              — Kaggle inference notebooks
docs/                   — PRDs, specs, sequences-schema.json
```

## Hard Rules — Never Change Without Asking

1. `h_gzip_incremental` formula in scorer.py — changing it invalidates all EL_gzip results
2. Output-transducer emit logic in tm_simulator.py — changing it invalidates all sequences
3. Tokenizer gate must run before any benchmark — never skip
4. `outputs.logits[:, -1, :]` is the only valid logit extraction — never use `model.generate()`
5. EL_gzip and SG are never described as the same metric anywhere in code or output

## Week 1 Gotchas

- Sequences are ASCII strings of "0"/"1" — NOT packed binary, NOT numpy arrays
- Discard (never truncate) TM runs that hit 10,000 transitions before 200 symbols
- Log every negative gzip delta — never silently clamp to zero
- Save invalid-token mass BEFORE renormalization — log it per model per run
- If Llama 3.2 access blocked on Kaggle after 30 min, switch to Phi-3-mini immediately
- Checkpoint results every 50 sequences — Kaggle sessions disconnect

## Planning Files (read-only reference)

Full specs live in the portfolio planning folder — do not modify them from this repo:
`E:\application\MS CS\portfolio-projects\00-solomonoff-llm\`
Files to read before implementing: FINAL_VULNERABILITY_SCAN.md → DESIGN.md → CLAUDE_CODE_PROMPT.md → tasks/todo.md
