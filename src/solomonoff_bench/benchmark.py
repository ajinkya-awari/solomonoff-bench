"""Main benchmark runner for Week 1 EL_gzip pilot.

Resumable: saves a checkpoint CSV every CHECKPOINT_EVERY sequences and
appends to benchmark_log.jsonl. On restart, already-scored sequence/config
combinations for the requested model are skipped.

Usage — call run_benchmark() directly from Python or a notebook:

    from pathlib import Path
    from solomonoff_bench.benchmark import run_benchmark
    run_benchmark(
        model_name="Qwen/Qwen2.5-3B",
        dataset_path=Path("data/sequences_mvp.json"),
        output_dir=Path("results/"),
    )

Note: there is no CLI entrypoint. The module docstring previously showed
`python -m solomonoff_bench.benchmark --model ...` but that was aspirational
and never implemented. Use the function API above.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Optional

CHECKPOINT_EVERY = 50
CONTEXT_LEN = 100
PREDICT_LEN = 100


def _dataset_sha256(dataset_path: Path) -> str:
    """Return the SHA-256 digest of the exact dataset file used for a run."""
    digest = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completion_key(
    sequence_id: str,
    model_name: str,
    dataset_sha256: str,
    context_len: int,
    predict_len: int,
) -> str:
    """Build a stable, model- and configuration-scoped completion key."""
    return json.dumps(
        {
            "sequence_id": sequence_id,
            "model": model_name,
            "dataset_sha256": dataset_sha256,
            "context_len": context_len,
            "predict_len": predict_len,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def load_dataset(dataset_path: Path) -> list[dict]:
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["records"]


def load_completed_ids(
    log_path: Path,
    *,
    model_name: str,
    dataset_sha256: str,
    context_len: int,
    predict_len: int,
) -> set[str]:
    """Read completed sequence IDs matching the requested run configuration.

    Current entries must carry a completion key (or equivalent metadata) for
    exact matching. Older Week 1/2 entries predate those fields; they remain
    resumable only when their recorded model matches, with an explicit warning
    because their dataset and window cannot be verified retrospectively.
    """
    completed: set[str] = set()
    legacy_model_only: set[str] = set()
    legacy_unscoped: set[str] = set()
    if not log_path.exists():
        return completed
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("status") != "done":
                    continue
                sequence_id = entry.get("sequence_id")
                if not isinstance(sequence_id, str):
                    continue

                expected_key = _completion_key(
                    sequence_id=sequence_id,
                    model_name=model_name,
                    dataset_sha256=dataset_sha256,
                    context_len=context_len,
                    predict_len=predict_len,
                )
                if entry.get("completion_key") == expected_key:
                    completed.add(sequence_id)
                    continue

                metadata_fields = {
                    "model": entry.get("model"),
                    "dataset_sha256": entry.get("dataset_sha256"),
                    "context_len": entry.get("context_len"),
                    "predict_len": entry.get("predict_len"),
                }
                if all(value is not None for value in metadata_fields.values()):
                    if (
                        metadata_fields["model"] == model_name
                        and metadata_fields["dataset_sha256"] == dataset_sha256
                        and metadata_fields["context_len"] == context_len
                        and metadata_fields["predict_len"] == predict_len
                    ):
                        completed.add(sequence_id)
                    continue

                # Legacy entries contain model but no dataset/window metadata.
                if entry.get("model") == model_name:
                    completed.add(sequence_id)
                    legacy_model_only.add(sequence_id)
                elif "model" not in entry:
                    legacy_unscoped.add(sequence_id)
            except json.JSONDecodeError:
                continue

    if legacy_model_only:
        warnings.warn(
            f"Reusing {len(legacy_model_only)} legacy completion entries matched "
            "by model only; dataset/window metadata was unavailable.",
            RuntimeWarning,
            stacklevel=2,
        )
    if legacy_unscoped:
        warnings.warn(
            f"Ignoring {len(legacy_unscoped)} legacy completion entries without "
            "a model; they cannot be safely attributed to this run.",
            RuntimeWarning,
            stacklevel=2,
        )
    return completed


def append_log(log_path: Path, entry: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def run_benchmark(
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    load_in_4bit: bool = False,
    context_len: int = CONTEXT_LEN,
    predict_len: int = PREDICT_LEN,
    checkpoint_every: int = CHECKPOINT_EVERY,
    verbose: bool = True,
) -> Path:
    """Run the full EL_gzip benchmark for one model.

    Returns path to the final results CSV.
    """
    from solomonoff_bench.models.local_model import LocalHFScorer
    from solomonoff_bench.baselines.gzip_baseline import GzipBaseline
    from solomonoff_bench.metrics.excess_loss import compute_el_gzip

    output_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize model name for filenames
    model_slug = model_name.replace("/", "--").replace(":", "-")
    csv_path = output_dir / ("el_gzip_" + model_slug + ".csv")
    log_path = output_dir / "benchmark_log.jsonl"
    checkpoint_path = output_dir / ("checkpoint_" + model_slug + ".json")

    records = load_dataset(dataset_path)
    dataset_sha256 = _dataset_sha256(dataset_path)
    completed_ids = load_completed_ids(
        log_path,
        model_name=model_name,
        dataset_sha256=dataset_sha256,
        context_len=context_len,
        predict_len=predict_len,
    )

    run_metadata = {
        "model": model_name,
        "dataset_sha256": dataset_sha256,
        "context_len": context_len,
        "predict_len": predict_len,
    }

    if verbose:
        print("Model:", model_name)
        print("Dataset:", dataset_path, "—", len(records), "sequences")
        print("Already done:", len(completed_ids), "/ resuming from checkpoint")

    # Load model
    scorer = LocalHFScorer(
        model_name_or_path=model_name,
        load_in_4bit=load_in_4bit,
    )

    # 5-prompt toy validation gate — MUST pass before full run
    toy_seqs = [r["sequence"] for r in records[:5]]
    gate_result = scorer.validate_toy_sequences(toy_seqs)
    append_log(log_path, {"event": "tokenizer_gate", "result": gate_result,
                          **run_metadata})
    if verbose:
        print("Tokenizer gate passed.")
        print("  Mean valid binary mass:", round(gate_result["mean_valid_binary_mass"], 4))
        print("  Mean invalid mass:", round(gate_result["mean_invalid_mass"], 4))

    gzip_baseline = GzipBaseline()

    # Write CSV header (or append if resuming)
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "sequence_id", "complexity_level", "n_states", "program_bits",
        "model", "context_len", "predict_len",
        "h_model_bits_per_sym", "h_gzip_bits_per_sym", "h_gzip_is_negative",
        "el_gzip", "mean_invalid_mass", "mean_valid_binary_mass",
    ])
    if write_header:
        writer.writeheader()

    done_this_run = 0
    t0 = time.time()

    for rec in records:
        sid = rec["sequence_id"]
        if sid in completed_ids:
            continue

        sequence = rec["sequence"]
        assert len(sequence) == context_len + predict_len, (
            "Sequence " + sid + " length mismatch"
        )

        # Score the prediction window
        window_results = scorer.score_sequence_window(
            sequence, context_len=context_len, predict_len=predict_len
        )
        p0_list = [r["p0_renorm"] for r in window_results]
        mean_invalid = sum(r["invalid_mass"] for r in window_results) / len(window_results)
        mean_valid = sum(r["valid_binary_mass"] for r in window_results) / len(window_results)

        # Compute EL_gzip
        el = compute_el_gzip(sequence, p0_list, context_len, predict_len)

        row = {
            "sequence_id": sid,
            "complexity_level": rec["complexity_level"],
            "n_states": rec["n_states"],
            "program_bits": rec["program_bits"],
            "model": model_name,
            "context_len": context_len,
            "predict_len": predict_len,
            "h_model_bits_per_sym": round(el["h_model_bits_per_sym"], 6),
            "h_gzip_bits_per_sym": round(el["h_gzip_bits_per_sym"], 6),
            "h_gzip_is_negative": el["h_gzip_is_negative"],
            "el_gzip": round(el["el_gzip"], 6),
            "mean_invalid_mass": round(mean_invalid, 6),
            "mean_valid_binary_mass": round(mean_valid, 6),
        }
        writer.writerow(row)
        csv_file.flush()

        append_log(log_path, {
            "status": "done",
            "sequence_id": sid,
            "completion_key": _completion_key(
                sequence_id=sid,
                model_name=model_name,
                dataset_sha256=dataset_sha256,
                context_len=context_len,
                predict_len=predict_len,
            ),
            **run_metadata,
            "el_gzip": el["el_gzip"],
            "h_gzip_is_negative": el["h_gzip_is_negative"],
        })
        completed_ids.add(sid)
        done_this_run += 1

        # Checkpoint every N sequences
        if done_this_run % checkpoint_every == 0:
            elapsed = time.time() - t0
            rate = done_this_run / elapsed
            remaining = (len(records) - len(completed_ids)) / max(rate, 1e-6)
            if verbose:
                print(
                    "  Checkpoint: done=" + str(done_this_run)
                    + " elapsed=" + str(round(elapsed, 1)) + "s"
                    + " rate=" + str(round(rate, 2)) + "/s"
                    + " ETA=" + str(round(remaining / 60, 1)) + "m"
                )
            with open(checkpoint_path, "w", encoding="utf-8") as cp:
                json.dump({"done": done_this_run, "total": len(records)}, cp)

    csv_file.close()

    # Final gzip diagnostics
    gzip_diag = gzip_baseline.diagnostics()
    append_log(log_path, {"event": "gzip_diagnostics", **run_metadata, **gzip_diag})

    if verbose:
        elapsed = time.time() - t0
        print("Done. Total scored this run:", done_this_run)
        print("Elapsed:", round(elapsed, 1), "s")
        print("Results:", csv_path)

    return csv_path
