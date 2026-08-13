"""Regression tests for model/config-scoped benchmark resume state."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solomonoff_bench.benchmark import _completion_key, load_completed_ids


MODEL = "Qwen/Qwen2.5-3B"
DATASET_SHA256 = "a" * 64


def _write_log(path: Path, entries: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def _new_done_entry(sequence_id: str, **overrides: object) -> dict:
    values = {
        "status": "done",
        "sequence_id": sequence_id,
        "model": MODEL,
        "dataset_sha256": DATASET_SHA256,
        "context_len": 100,
        "predict_len": 100,
    }
    values.update(overrides)
    values["completion_key"] = _completion_key(
        sequence_id=sequence_id,
        model_name=values["model"],
        dataset_sha256=values["dataset_sha256"],
        context_len=values["context_len"],
        predict_len=values["predict_len"],
    )
    return values


def test_resume_state_is_scoped_to_model_dataset_and_window(tmp_path: Path):
    log_path = tmp_path / "benchmark_log.jsonl"
    _write_log(
        log_path,
        [
            _new_done_entry("matching"),
            _new_done_entry("other-model", model="Qwen/Qwen2.5-1.5B"),
            _new_done_entry("other-dataset", dataset_sha256="b" * 64),
            _new_done_entry("other-window", predict_len=50),
        ],
    )

    completed = load_completed_ids(
        log_path,
        model_name=MODEL,
        dataset_sha256=DATASET_SHA256,
        context_len=100,
        predict_len=100,
    )

    assert completed == {"matching"}


def test_legacy_entries_remain_resumable_only_for_their_model(tmp_path: Path):
    log_path = tmp_path / "benchmark_log.jsonl"
    _write_log(
        log_path,
        [{"status": "done", "sequence_id": "legacy", "model": MODEL}],
    )

    with pytest.warns(RuntimeWarning, match="legacy"):
        completed = load_completed_ids(
            log_path,
            model_name=MODEL,
            dataset_sha256=DATASET_SHA256,
            context_len=100,
            predict_len=100,
        )
    assert completed == {"legacy"}

    assert load_completed_ids(
        log_path,
        model_name="Qwen/Qwen2.5-1.5B",
        dataset_sha256=DATASET_SHA256,
        context_len=100,
        predict_len=100,
    ) == set()
