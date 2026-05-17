"""Tests for the CLI (subprocess integration)."""
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable
REPO = str(Path(__file__).parent.parent)


def test_cli_run_produces_output():
    result = subprocess.run(
        [PYTHON, "-m", "wcsim.cli", "run", "-n", "10", "--seed", "42", "--workers", "1"],
        capture_output=True, text=True, timeout=60, cwd=REPO,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Win" in result.stdout or "%" in result.stdout


def test_cli_match_prints_probabilities():
    result = subprocess.run(
        [PYTHON, "-m", "wcsim.cli", "match", "Brazil", "France", "--neutral"],
        capture_output=True, text=True, timeout=10, cwd=REPO,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "wins" in result.stdout.lower()


def test_cli_version():
    result = subprocess.run(
        [PYTHON, "-m", "wcsim.cli", "version"],
        capture_output=True, text=True, timeout=5, cwd=REPO,
    )
    assert result.returncode == 0
    assert "wcsim" in result.stdout.lower()
