"""Async git commit+push helper for user-submitted insults.

Failures are logged but never raised - the kiosk must not crash if the
network is down or git misconfigured.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path


def _run_sync(repo_dir: Path, message: str) -> None:
    try:
        subprocess.run(
            ["git", "add", "insults/submissions/"],
            cwd=repo_dir, check=True, timeout=15,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_dir, check=True, timeout=15,
            capture_output=True,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_dir, check=True, timeout=30,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[github_sync] push failed: {e}")


def push_submission_async(repo_dir: Path, message: str) -> threading.Thread:
    t = threading.Thread(
        target=_run_sync, kwargs={"repo_dir": repo_dir, "message": message},
        daemon=True,
    )
    t.start()
    return t
