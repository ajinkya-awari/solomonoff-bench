# SolomonoffBench

**Empirical benchmark measuring how many extra bits per symbol LLMs waste compared to the theoretically optimal predictor (Solomonoff induction), as a function of formally computed Kolmogorov complexity.**

[![arXiv](https://img.shields.io/badge/arXiv-coming--soon-b31b1b)](https://arxiv.org)
[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/ajinkya-awari/solomonoff-bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen)](tests/)
[![GitHub](https://img.shields.io/badge/GitHub-ajinkya--awari%2Fsolomonoff--bench-181717?logo=github)](https://github.com/ajinkya-awari/solomonoff-bench)

---

## Hero Figure

> **Figure 1 (Week 1 Pilot)** — EL_gzip vs TM program complexity for two local LLMs.
> Results from Kaggle T4 GPU inference; figure generated after benchmark run.

![Figure 1 — EL_gzip pilot](results/figures/fig1_mvp_el_gzip.png)

*Note: EL_gzip is the Week 1 pilot metric (gzip-normalized excess code length). The CTW-normalized Solomonoff Gap (SG) is the Week 2+ research metric. They are not the same quantity and are never plotted on the same axis.*

---

## What This Is

AIXI requires Solomonoff induction as its predictive component — but Solomonoff induction is incomputable. Modern LLMs are the best practical sequence predictors available today. This project asks:

> **Can free, reproducible LLMs approximate the universal prediction behaviour required by AIXI-like agents on program-generated binary sequences of increasing complexity?**

The **Solomonoff Gap (SG)** metric quantifies exactly how many bits per symbol an LLM wastes over the best practical Solomonoff approximator (CTW), as a function of formally computed Kolmogorov complexity K.

**Week 1 pilot:** 300 TM-generated binary sequences, 4 complexity levels, 2 local models, gzip-normalized `EL_gzip` metric.  
**Week 2+:** CTW-normalized `SG`, full 2,400-sequence dataset, arXiv paper.

---

## Key Results (Week 1 Pilot)

| Model | Level 1 (24 bits) | Level 2 (40 bits) | Level 3 (50 bits) | Level 4 (60 bits) |
|-------|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| Phi-3 Mini (3.8B) | — | — | — | — |
| Llama 3.2 (3B) | — | — | — | — |

*Results pending Kaggle T4 benchmark run. See `notebooks/03_benchmark_local_kaggle.ipynb`.*

---

## Repository Structure

```
solomonoff-bench/
├── src/solomonoff_bench/
│   ├── sequences/          # TM simulator + sequence generator
│   ├── models/             # Tokenizer validation + local HF scorer
│   ├── baselines/          # Incremental gzip + random baselines
│   ├── metrics/            # EL_gzip computation
│   ├── analysis/           # Plots + stats + mock results
│   └── benchmark.py        # Resumable main runner
├── notebooks/
│   ├── 01_generate_sequences.ipynb
│   ├── 02_validate_tokenizers.ipynb
│   ├── 03_benchmark_local_kaggle.ipynb   ← run on Kaggle T4
│   └── 04_analysis_and_plots.ipynb
├── data/sequences_mvp.json              ← gitignored (300 sequences)
├── results/
│   ├── el_gzip_results.csv
│   ├── figures/fig1_mvp_el_gzip.png
│   └── minimum_viable_result_check.txt
└── tests/                               ← 43 tests, all passing
```

---

## Quickstart

```bash
git clone https://github.com/ajinkya-awari/solomonoff-bench
cd solomonoff-bench
pip install -e ".[dev]"

# Generate 300 sequences (CPU, ~1s)
python src/solomonoff_bench/sequences/generate_sequences.py

# Run tests
python -m pytest tests/ -v

# Full benchmark (needs Kaggle T4 — see notebooks/03_benchmark_local_kaggle.ipynb)
```

---

## Ten Non-Negotiable Rules

1. TM output-transducer convention — written symbol emitted at every transition
2. Sequences are ASCII strings of `"0"`/`"1"` characters — never packed binary
3. Incremental gzip only: `8*(len(gzip(prefix+suffix)) - len(gzip(prefix))) / len(suffix)`
4. Negative gzip deltas are logged and reported — never silently clamped
5. Invalid-token mass saved **before** renormalization in every scorer run
6. Direct `model(**inputs).logits[:, -1, :]` forward pass — never `model.generate(output_scores=True)`
7. 5-prompt toy gate must pass before any full benchmark run
8. `EL_gzip` ≠ Solomonoff Gap — never describe or plot them as the same metric
9. Week 1 results never claim CTW-normalized SG
10. All infrastructure runs at £0 — Kaggle T4 for GPU, no paid APIs

---

## Citation

```bibtex
@misc{awari2026solomonoff,
  title   = {Can Free LLMs Serve as Practical Context Models for AIXI?
             A Pilot Benchmark on Program-Generated Binary Sequences},
  author  = {Awari, Ajinkya},
  year    = {2026},
  note    = {Preprint. arXiv:coming-soon}
}
```

---

## Related Work

- Delétang et al. 2023 — *Language Modeling Is Compression* (arXiv:2309.10668) — closest prior work; our K-conditioned design asks a fundamentally different question
- Hutter 2000 — *A Theory of Universal Artificial Intelligence Based on Algorithmic Complexity*
- Willems, Shtarkov & Tjalkens 1995 — *The Context-Tree Weighting Method* (CTW baseline)
