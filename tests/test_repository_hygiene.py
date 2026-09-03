from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_upload_candidate_contains_no_dataset_files() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_repository.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

