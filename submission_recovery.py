#!/usr/bin/env python3
"""Keep user-submitted insults alive across the boot-time `git reset --hard`.

start.sh resets the Pi's checkout to origin/main on every boot, which throws
away any submission whose push never reached GitHub (network down, missing
credentials, or a rejected push). start.sh therefore runs:

    python3 submission_recovery.py save    <repo> <stash>   # before fetch/reset
    python3 submission_recovery.py restore <repo> <stash>   # after the reset

`save` stashes rows in insults/submissions/*.csv that aren't in origin/main
(as last fetched, i.e. what we know reached GitHub). `restore` appends any
stashed rows the fresh checkout lacks, commits, and pushes via github_sync.
If that push fails the commit stays local, so the next boot's `save` finds
the rows again. Stdlib only; never raises (a boot must always launch the app).
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path

from github_sync import _run_sync

SUBMISSION_FILES = ("insults/submissions/adjectives.csv", "insults/submissions/nouns.csv")
HEADER = "word,category,submission_date"
ROW_END = "\r\n"   # insult_store writes with csv.writer's default terminator


def _rows(text: str) -> list[str]:
    """Data rows (header and blank lines dropped), line endings stripped."""
    lines = text.splitlines()
    return [ln for ln in lines[1:] if ln.strip()]


def unpushed_rows(local_text: str, base_text: str) -> list[str]:
    """Rows in local that base doesn't have (multiset difference, order kept)."""
    base = Counter(_rows(base_text))
    out = []
    for row in _rows(local_text):
        if base[row] > 0:
            base[row] -= 1
        else:
            out.append(row)
    return out


def missing_rows(current_text: str, rows) -> list[str]:
    present = set(_rows(current_text))
    return [r for r in rows if r not in present]


def append_rows(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_bytes() if path.exists() else b""
    with path.open("ab") as fh:
        if not existing:
            fh.write((HEADER + ROW_END).encode())
        elif not existing.endswith(b"\n"):
            fh.write(ROW_END.encode())
        fh.write("".join(r + ROW_END for r in rows).encode())


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _show(repo: Path, ref: str, rel: str) -> str:
    try:
        return subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=repo, check=True,
                              capture_output=True, text=True, timeout=15).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""


def save(repo, stash, ref="origin/main") -> int:
    """Stash rows not yet on `ref`. Returns how many rows are stashed in total."""
    repo, stash = Path(repo), Path(stash)
    total = 0
    try:
        for rel in SUBMISSION_FILES:
            local = _read(repo / rel)
            if not local:
                continue
            rows = unpushed_rows(local, _show(repo, ref, rel))
            stash_file = stash / Path(rel).name
            # Merge with anything an interrupted earlier boot left behind.
            rows = missing_rows(_read(stash_file), rows)
            if rows:
                append_rows(stash_file, rows)
            total += len(_rows(_read(stash_file)))
        print(f"[submissions] save: {total} unpushed row(s) stashed", flush=True)
    except Exception as e:  # noqa: BLE001 — never block the boot
        print(f"[submissions] save failed: {type(e).__name__}: {e}", flush=True)
    return total


def restore(repo, stash) -> int:
    """Re-append stashed rows missing from the checkout, then commit + push."""
    repo, stash = Path(repo), Path(stash)
    added = []
    try:
        for rel in SUBMISSION_FILES:
            stash_file = stash / Path(rel).name
            rows = _rows(_read(stash_file))
            if not rows:
                continue
            rows = missing_rows(_read(repo / rel), rows)
            if rows:
                append_rows(repo / rel, rows)
                added += rows
        if added:
            words = ", ".join(r.split(",", 1)[0] for r in added)
            ok = _run_sync(repo_dir=repo, message=f"recovered submissions: {words}")
            print(f"[submissions] restore: re-added {len(added)} row(s) ({words}); "
                  f"push {'ok' if ok else 'FAILED - will retry next boot'}", flush=True)
        else:
            print("[submissions] restore: nothing to re-add", flush=True)
        # Rows now live in the checkout (committed even if the push failed), so
        # the next boot's save() rediscovers anything still unpushed.
        for rel in SUBMISSION_FILES:
            (stash / Path(rel).name).unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001 — never block the boot
        print(f"[submissions] restore failed: {type(e).__name__}: {e}", flush=True)
    return len(added)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 3 or argv[0] not in ("save", "restore"):
        print("usage: submission_recovery.py save|restore <repo> <stash>")
        return 0
    cmd, repo, stash = argv
    (save if cmd == "save" else restore)(repo, stash)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
