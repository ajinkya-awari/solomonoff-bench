# CLAUDE.md — solomonoff-bench

## Stack and Commands

**Repo:** `E:\application\MS CS\solomonoff-bench\` — installable Python package

**Current handoff:** Read `IMPLEMENTATION_AUDIT_2026-08-13.md` before making changes. It records the five completed local fixes, verification evidence, untouched working-tree items, and the remaining Project 00 work.

**Stack:** Python 3.10 · PyTorch · HuggingFace Transformers · gzip · ruff · pytest

**Primary models:** `Qwen/Qwen2.5-3B` + `Qwen/Qwen2.5-1.5B` (both base, ungated) on Kaggle T4
**Rejected models:** Phi-3-mini (rope_scaling KeyError on Kaggle transformers), Llama 3.2 3B (gated — license form inaccessible), TinyLlama/Qwen-Instruct (chat-tuned, fail binary token gate)

**Commands:**
```
python -m pytest tests/ -v                          # run all 43 tests
python -m solomonoff_bench.sequences.generate_sequences  # build 300-seq dataset
python -m solomonoff_bench.benchmark  # NOTE: no CLI args — call run_benchmark() directly in Python
```

## Architecture (actual package structure)

```
src/solomonoff_bench/
  sequences/
    tm_simulator.py        — output-transducer TM, 200-symbol, 10K-step limit
    generate_sequences.py  — 300 sequences, 4 levels (3-6 states), 75/level, deduplicated
  models/
    base_model.py          — abstract scorer + 5-prompt toy validation gate
    tokenizer_validation.py — token IDs for "0"/"1" variants, invalid-mass logging
    local_model.py         — direct logits scorer (NOT model.generate)
  baselines/
    gzip_baseline.py       — incremental gzip, negative-delta tracking
    random_baseline.py     — P=0.5, H=1.0 bits/sym
  metrics/
    excess_loss.py         — EL_gzip = H(model) - H_gzip_incremental (NOT SG)
  analysis/
    plots.py               — Figure 1 (EL_gzip vs program_bits, bootstrap CI)
    stats.py               — level_summary, minimum_viable_result_check
    mock_results.py        — synthetic CSV for pipeline smoke testing
  benchmark.py             — resumable runner, checkpoint every 50 seqs
notebooks/
  01_generate_sequences.ipynb
  02_validate_tokenizers.ipynb
  03_benchmark_local_kaggle.ipynb   ← run on Kaggle T4 for Day 3
  04_analysis_and_plots.ipynb
tests/
  test_tm_simulator.py  — 20 tests (all passing)
  test_baselines.py     — 13 tests (all passing)
  test_metrics.py       — 10 tests (all passing)
data/sequences_mvp.json             — gitignored, 300 seqs generated
results/el_gzip_results.csv         — pending Kaggle run
results/figures/fig1_mvp_el_gzip.png — generated after benchmark
```

## Implementation Status

| Day | Task | Status |
|-----|------|--------|
| 1 | TM simulator + 300 sequences | ✅ Done |
| 2 | Baselines + tokenizer validation + scorer | ✅ Done |
| 2+ | Analysis pipeline, notebooks, README | ✅ Done |
| 3 | Kaggle T4 benchmark run | ⬜ NEXT — run 03_benchmark_local_kaggle.ipynb |
| 4 | Figure 1 + minimum_viable_result_check | ⬜ After Day 3 results |
| 5 | 4-page preprint on Overleaf | ⬜ |
| 6–7 | AIXI Labs fellowship submission | ⬜ |

## Hard Rules — Never Violate

1. Output-transducer convention — emit WRITTEN symbol at every transition
2. Sequences are ASCII "0"/"1" strings — NOT packed binary, NOT numpy arrays
3. `h_gzip_incremental` formula only — never whole-string ratio
4. Log every negative gzip delta — never silently clamp
5. Save invalid-token mass BEFORE renormalization
6. `outputs.logits[:, -1, :]` only — never `model.generate(output_scores=True)`
7. 5-prompt toy gate must pass before full benchmark
8. EL_gzip ≠ SG — never describe or plot as the same metric
9. Week 1 outputs never claim CTW-normalized Solomonoff Gap

## Planning Files (read-only)

`E:\application\MS CS\portfolio-projects\00-solomonoff-llm\`
Read order: FINAL_VULNERABILITY_SCAN.md → DESIGN.md → CLAUDE_CODE_PROMPT.md → tasks/todo.md
