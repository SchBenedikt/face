"""config must stay importable when python-dotenv is missing or broken.

The real import is replaced by a PYTHONPATH shim in a subprocess, so this does
not depend on whether python-dotenv is installed in the test environment.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_config_with_dotenv_shim(tmp_path: Path, shim_source: str) -> subprocess.CompletedProcess:
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    (shim_dir / "dotenv.py").write_text(shim_source, encoding="utf-8")

    env = os.environ.copy()
    env.pop("FACE_DATA_DIR", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(shim_dir), str(PROJECT_ROOT), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["PYTHONNOUSERSITE"] = "1"

    script = "import config; print(config.DATA_DIR); print(config.VECTOR_DB_PATH)"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _assert_defaults_without_traceback(completed: subprocess.CompletedProcess) -> None:
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output[-2000:]
    assert "Traceback" not in output, output[-2000:]
    assert str(PROJECT_ROOT / "data") in completed.stdout, output[-2000:]
    assert str(PROJECT_ROOT / "data" / "face_vectors.db") in completed.stdout, output[-2000:]


def test_requirements_declares_python_dotenv():
    requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    packages = [
        line.split("#", 1)[0].strip()
        for line in requirements.splitlines()
        if line.split("#", 1)[0].strip()
    ]

    assert any(package.startswith("python-dotenv") for package in packages), packages


def test_missing_dotenv_package_falls_back_to_defaults(tmp_path):
    completed = _run_config_with_dotenv_shim(
        tmp_path, 'raise ImportError("python-dotenv blocked for this test")\n'
    )

    _assert_defaults_without_traceback(completed)


def test_broken_dotenv_loader_falls_back_to_defaults(tmp_path):
    completed = _run_config_with_dotenv_shim(
        tmp_path,
        "def load_dotenv(*args, **kwargs):\n"
        "    raise RuntimeError('broken .env file')\n",
    )

    _assert_defaults_without_traceback(completed)
