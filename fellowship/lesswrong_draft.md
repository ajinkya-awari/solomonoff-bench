# LessWrong Post Draft
**Title:** We Measured How Far Free LLMs Are From Solomonoff Induction. The Answer Is: Very Far.  
**Tags:** AI, AIXI, Solomonoff, Benchmarks, Kolmogorov Complexity  
**Target:** 900–1,100 words  

---

## We Measured How Far Free LLMs Are From Solomonoff Induction. The Answer Is: Very Far.

AIXI is the theoretical gold standard for intelligent agents. At its core sits Solomonoff induction — the provably optimal sequence predictor. The problem: Solomonoff induction is incomputable. You cannot run it on a computer.

Modern LLMs are the most capable practical sequence predictors we have. A natural question follows: can they substitute for Solomonoff induction inside AIXI-like systems? If yes, you get a practical universal agent. If no, you want to know by how much they fall short — and whether that gap grows as the input gets more algorithmically complex.

Last week I built a benchmark to start answering this. Here is what I found, what it means, and where the research goes next.

---

### The setup

I generated 300 binary sequences using Turing machines with formally computed program lengths — 75 sequences each at four complexity levels (24, 40, 50, and 60 program bits). The key feature: complexity is formally defined, not estimated. Each sequence comes from a TM whose full program is known and whose bit length is computable exactly.

Then I ran two free Qwen2.5 base models on these sequences using a Kaggle T4 GPU, extracting exact next-token probabilities via direct forward-pass logits (not sampling, not generation scores — actual conditional probabilities at each position).

The metric for Week 1 is **EL_gzip**: the per-symbol gap between what the LLM predicts and what incremental gzip compression achieves on the same sequence. Think of it as a lower-bound proxy for the true Solomonoff Gap — how much worse is the model than even a trivial compressor, let alone the theoretical optimum?

The scale works like this:
- EL_gzip = 0 → model matches gzip (already impressively low bar)
- EL_gzip = 1.0 → model is no better than a coin flip
- EL_gzip > 1.0 → model is worse than a coin flip

---

### The result

Neither model comes close to gzip.

**Qwen2.5-3B** (3 billion parameters, free, runs on a single GPU):
- EL_gzip = 0.917 to 0.972 bits/sym across all four complexity levels
- Better than random prediction (EL_gzip < 1.0)
- 0.94 bits/sym above gzip on average

**Qwen2.5-1.5B** (1.5 billion parameters):
- EL_gzip = 1.028 to 1.096 bits/sym
- Worse than random prediction at every complexity level

Gzip — a 30-year-old compression algorithm — comprehensively outpredicts both models on these algorithmically structured sequences.

![EL_gzip vs program bits](https://github.com/ajinkya-awari/solomonoff-bench/blob/master/paper/figures/fig1_mvp_el_gzip.png?raw=true)

There is a second finding that is arguably more important than the raw scores: **both models show near-flat profiles across all four complexity levels**. The spread from lowest to highest EL_gzip across the 24–60 bit range is only 0.055 bits/sym for the 3B model and 0.068 bits/sym for the 1.5B model. Given that program lengths span a 2.5× range, the absence of any monotonic increase is striking.

This is not a positive result for LLMs. A model that is genuinely tracking algorithmic complexity should show increasing prediction difficulty as the generating program gets longer. Neither model does.

---

### What this does and does not prove

**What it proves:**

Free local LLMs are poor approximations to Solomonoff induction on program-generated binary sequences, performing well below even a trivial gzip baseline. The gap is not small — the 3B model uses nearly 1 full bit per symbol more than gzip on average, which represents essentially no algorithmic understanding of the sequences.

**What it does not prove:**

The EL_gzip metric is too coarse to be the final word. Incremental gzip on 100-character ASCII binary strings trivially achieves near-zero compression cost regardless of the sequence's Kolmogorov complexity — DEFLATE finds run-length and Huffman patterns that have nothing to do with the sequence's algorithmic structure. The flat EL_gzip profiles may be a limitation of the baseline as much as of the models.

The theoretically correct baseline is **Context Tree Weighting (CTW)** — a parameter-free algorithm that achieves provably near-optimal per-symbol predictions for binary sequences, and is the best known practical approximation to Solomonoff induction. The true Solomonoff Gap is SG(M, x) = H(M, x) − H_CTW(x). Week 2 implements CTW and computes actual SG values.

---

### Why gzip gives a flat profile and CTW will not

Gzip on a 100-character window has a granularity problem. It compresses most short ASCII binary strings to near-zero additional bytes once it has the 100-char prefix context, because DEFLATE's Huffman tables work at the byte level and 100 bytes is enough to learn any short repeating pattern. As a result, the incremental gzip cost is approximately zero for almost every sequence regardless of its true Kolmogorov complexity.

CTW does not have this problem. It produces per-symbol probability estimates with provable redundancy bounds, making it sensitive to the actual structure (or lack thereof) in each sequence. If a model genuinely tracks algorithmic complexity, the CTW-normalized SG will show it. If the gap is flat with CTW too, that is the definitive answer: LLMs are simply insensitive to formal algorithmic complexity.

---

### The question that Week 2 will answer

At what program-bit threshold does SG exceed 0.5 bits/sym for each model? Call this K*. Below K*, the LLM is a reasonable (if imperfect) practical context model for an AIXI-like agent. Above K*, it is not.

If K* exists and is well-defined, it gives a formal, empirically grounded answer to the AIXI-LLM substitution question. If K* does not exist because SG is uniformly large at all complexity levels, that too is an answer — and a significant one for anyone building practical AIXI-inspired systems.

---

### Reproduce it yourself

Everything is free and public:

- Code + data: https://github.com/ajinkya-awari/solomonoff-bench
- Paper source (NeurIPS 2026 format): https://github.com/ajinkya-awari/solomonoff-bench/blob/master/paper/main.tex
- Runs on a free Kaggle T4 GPU notebook, zero API cost

The full 4-page preprint is linked in the repo. arXiv submission is in progress.

Week 2 (CTW implementation + full SG results) will be posted here when complete.

---

*Questions, corrections, and pushback welcome in the comments.*
