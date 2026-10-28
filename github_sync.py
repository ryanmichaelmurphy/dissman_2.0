"""Async git commit+push helper for user-submitted insults.

Failures are logged but never raised - the kiosk must not crash if the
network is down or git misconfigured.

The push rebases onto origin/main first: the Pi's checkout is usually behind
main (code is pushed from dev between reboots), and a plain push would be
rejected as non-fast-forward. A commit whose push still fails stays local;
submission_recovery.py (run by start.sh) re-pushes it on the next boot instead
of letting `git reset --hard origin/main` wipe it.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path


def _git(repo_dir, *args, timeout=30):
    return subprocess.run(
        ["git", *args], cwd=repo_dir, check=True, timeout=timeout,
        capture_output=True,
    )


def _stderr(e):
    err = getattr(e, "stderr", None)
    if isinstance(err, bytes):
        err = err.decode(errors="replace")
    return (err or "").strip()


def _rebase_and_push(repo_dir) -> None:
    """Rebase local commits onto origin/main and push; one retry if rejected."""
    for attempt in (1, 2):
        try:
            _git(repo_dir, "pull", "--rebase", "--autostash", "--quiet", "origin", "main")
        except subprocess.CalledProcessError:
            subprocess.run(["git", "rebase", "--abort"], cwd=repo_dir,
                           capture_output=True, timeout=15)
            raise
        try:
            _git(repo_dir, "push", "origin", "main")
            return
        except subprocess.CalledProcessError as e:
            if attempt == 2:
                raise
            print(f"[github_sync] push rejected, rebasing and retrying: {_stderr(e)}",
                  flush=True)


def _run_sync(repo_dir: Path, message: str) -> bool:
    """add + commit + rebase + push. Returns True if the push succeeded."""
    try:
        _git(repo_dir, "add", "insults/submissions/", timeout=15)
        try:
            _git(repo_dir, "commit", "-m", message, timeout=15)
        except subprocess.CalledProcessError as e:
            # Usually "nothing to commit"; still push any earlier local commits.
            print(f"[github_sync] commit skipped: {_stderr(e) or e}", flush=True)
        _rebase_and_push(repo_dir)
        print(f"[github_sync] pushed: {message}", flush=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[github_sync] push failed: {e} {_stderr(e)}", flush=True)
        return False


def push_submission_async(repo_dir: Path, message: str) -> threading.Thread:
    t = threading.Thread(
        target=_run_sync, kwargs={"repo_dir": repo_dir, "message": message},
        daemon=True,
    )
    t.start()
    return t
