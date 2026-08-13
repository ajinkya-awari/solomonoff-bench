# Project 00 Implementation Audit and Claude Code Handoff

**Audit date:** 2026-08-13
**Repository:** `E:\application\MS CS\solomonoff-bench\`
**Scope:** Local maintenance fixes 1–5 requested after the read-only audit.
**Authority:** This file records the current implementation handoff. It does not replace the portfolio planning files in `E:\application\MS CS\portfolio-projects\00-solomonoff-llm\`.

## What Was Completed

Five fixes were applied one at a time. Each change was reviewed with a diff before commit, verified locally, and committed without pushing.

| Fix | Change | Commit |
|---|---|---|
| 1 | Replaced the invalid setuptools backend `setuptools.backends.legacy:build` with `setuptools.build_meta`. | `eeff90f` |
| 2 | Replaced process-salted `hash(canary_type)` with a SHA-256-derived stable seed offset. Added a cross-`PYTHONHASHSEED` regression test. | `c1dfbe3` |
| 3 | Added `_default_output_path()` and corrected the module entrypoint to write `data/sequences_mvp.json` inside this repository. Added a path regression test. | `55728a9` |
| 4 | Corrected README scope: current SG results cover 300 TM sequences and 600 model-sequence pairs; the 2,400-record TM+CA+LFSR+canary dataset is available, but full SG computation is planned for Week 3. | `77584a2` |
| 5 | Scoped benchmark resume state by sequence ID, model, dataset SHA-256, context length, and prediction length. New log entries carry these fields and a completion key. Legacy entries with a matching model remain resumable with a warning; entries without a model are ignored safely. | `5c52c64` |

## Verification Evidence

The following commands were run after all five commits:

```text
pip install --dry-run --no-deps --no-build-isolation .
Would install solomonoff-bench-0.1.0

python -m pytest tests/test_ctw.py -q
13 passed

python -m pytest -q
93 passed

Python compilation
33 Python files, 0 failures
```

The committed diff from `origin/master` is clean. The branch is five commits ahead of `origin/master`; no push was performed.

No Kaggle benchmark was rerun. No Zenodo record, published paper, or Git history was modified or rewritten.

## Working-Tree Items Not Part of These Fixes

These pre-existing items were deliberately left untouched and require an explicit decision:

```text
M  fellowship/cover_letter_aixi_labs.md
?? bash.exe.stackdump
?? results/el_gzip_results.csv
```

The old parent-level file `E:\application\MS CS\data\sequences_mvp.json` also remains. Fix 3 stopped future writes there; it was not deleted because deletion was outside the requested scope.

## Current Project 00 State

- Week 1 TM dataset generation works locally and writes inside this repository.
- The extended dataset exists at `data/sequences_full.json` with 2,400 records: 300 TM, 600 cellular automata, 600 LFSR, and 900 canary records.
- Current SG results are still the 300 TM sequences × 2 models = 600 model-sequence pairs in `results/sg_ctw_results.csv`.
- The project is locally implemented and tested, but the local commits await the user’s review and manual push decision.

## Pending Work

### Required before public release

1. Review the five local commits and push them manually only after approval. Do not force-push.
2. Update the repository documentation that is now stale:
   - `CLAUDE.md` still says “43 tests,” says the Kaggle run is next, and lists later outputs as pending.
   - `README.md` still has a `tests-43-passing` badge.
   - Fellowship drafts and notebook comments still contain older “CTW is not yet implemented” or Week 1-only wording where applicable.
3. Decide whether to keep or remove the unrelated `bash.exe.stackdump` and untracked `results/el_gzip_results.csv`.
4. Decide whether the old parent-level dataset should be archived or removed in a separate, explicitly approved cleanup.

### Remaining engineering hardening

5. Review `src/solomonoff_bench/models/local_model.py` before any new model run:
   - pin model revisions for reproducibility;
   - avoid `trust_remote_code=True` unless the exact repository/revision has been reviewed;
   - make tokenizer truncation explicit and log or reject truncation.
6. Refactor the benchmark CSV writer to use a context manager or `try/finally` so interrupted runs close files safely.
7. Apply the new model/config-aware resume logic to the Kaggle notebook runner too; the notebook still has its own sequence-ID-only resume implementation.
8. Run Ruff in an environment where the declared `ruff` dependency is installed, then resolve any findings.

### Week 3 research work (not part of the five fixes)

9. Extend SG computation from the 300 TM sequences to the full 2,400-record dataset only after defining the analysis policy for repeated fixed-pattern canaries and cross-generator overlaps.
10. Re-run the result/paper consistency audit after any Week 3 computation. Do not describe full-dataset SG results before those results actually exist.

### Portfolio governance outside this repository

11. Reconcile the portfolio-level `PORTFOLIO_STATUS.md` and portfolio `CLAUDE.md`: they still contain the old “no implementation code” / “Day 3 pending” state and the portfolio-wide freeze language. This is outside the five local fixes and was intentionally not changed here.

## Safe Next Session Order

1. Read this file.
2. Read `CLAUDE.md` and the portfolio planning files in their declared order.
3. Check `git status` and preserve the three unrelated working-tree items.
4. Do not run Kaggle, modify Zenodo, or push until the user explicitly authorizes the specific action.
5. If implementing a pending item, make one focused change, show its diff, run its regression test, and commit locally.
