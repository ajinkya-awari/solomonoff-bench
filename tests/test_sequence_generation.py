"""Regression tests for the MVP dataset entrypoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from solomonoff_bench.sequences.generate_sequences import _default_output_path


def test_default_output_path_is_repository_data_directory():
    repo_root = Path(__file__).resolve().parents[1]
    expected = repo_root / "data" / "sequences_mvp.json"

    assert _default_output_path() == expected
    assert _default_output_path().parent == repo_root / "data"
