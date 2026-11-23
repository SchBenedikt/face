"""
Regression test: a missing/broken data directory must produce a readable
message in the UI instead of a traceback.

The app is executed in a subprocess so the process-wide FACE_DATA_DIR setting
of the normal test session does not leak in.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def _run_app_with_data_dir(data_dir: Path) -> subprocess.CompletedProcess:
    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(STUBS_DIR)!r})
import face_recognition  # noqa: F401 - resolved through the stub directory
import deepface  # noqa: F401

from streamlit.testing.v1 import AppTest

at = AppTest.from_file({str(APP_FILE)!r}, default_timeout=120)
at.run()
print("EXCEPTIONS:", [str(element.value) for element in at.exception])
print("ERRORS:", [str(element.value) for element in at.error])
"""
    env = {
        "FACE_DATA_DIR": str(data_dir),
        "PATH": "/usr/bin:/bin",
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": str(PROJECT_ROOT),
    }
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_broken_data_directory_shows_message_not_traceback(tmp_path):
    # A *file* where the data directory should be cannot be created as a dir.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")

    completed = _run_app_with_data_dir(blocked)
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output[-2000:]
    assert "EXCEPTIONS: []" in output, output[-2000:]
    assert "The vector database could not be opened." in output, output[-2000:]
