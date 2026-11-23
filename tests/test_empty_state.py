"""A fresh install with an empty data directory must boot with guidance.

This runs in a subprocess so the data directory is guaranteed to be new and is
not populated by the shared fixtures of the main UI test session.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUBS_DIR = PROJECT_ROOT / "tests" / "stubs"
APP_FILE = PROJECT_ROOT / "app.py"


def test_empty_data_directory_boots_with_actionable_message(tmp_path):
    data_dir = tmp_path / "fresh-data"
    assert not data_dir.exists()

    script = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
sys.path.insert(0, {str(STUBS_DIR)!r})
import face_recognition  # noqa: F401 - resolved through the stub directory
import deepface  # noqa: F401

from pathlib import Path
from streamlit.testing.v1 import AppTest

at = AppTest.from_file({str(APP_FILE)!r}, default_timeout=180)
at.run()
print("EXCEPTIONS:", [str(element.value) for element in at.exception])
print("INFOS:", [str(element.value) for element in at.info])
print("DATA_EXISTS:", Path({str(data_dir)!r}).is_dir())
print("DB_EXISTS:", (Path({str(data_dir)!r}) / "face_vectors.db").exists())
"""
    env = {
        "FACE_DATA_DIR": str(data_dir),
        "PATH": "/usr/bin:/bin",
        "ANONYMIZED_TELEMETRY": "False",
        "PYTHONPATH": str(PROJECT_ROOT),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output[-3000:]
    assert "EXCEPTIONS: []" in output, output[-3000:]
    assert "Your database is empty" in output, output[-3000:]
    assert "Import images" in output, output[-3000:]
    assert "Web import" in output, output[-3000:]
    assert "DATA_EXISTS: True" in output, output[-3000:]
    assert "DB_EXISTS: True" in output, output[-3000:]
