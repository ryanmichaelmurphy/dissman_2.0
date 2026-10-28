import subprocess

import pytest

from submission_recovery import (
    SUBMISSION_FILES, append_rows, missing_rows, restore, save, unpushed_rows,
)

HEADER = "word,category,submission_date"


def _csv(*rows, crlf=True):
    nl = "\r\n" if crlf else "\n"
    return nl.join([HEADER, *rows]) + nl


# --- pure helpers ------------------------------------------------------------

def test_unpushed_rows_are_local_minus_base():
    base = _csv("butt,r,2026-05-19T01:30:29")
    local = _csv("butt,r,2026-05-19T01:30:29", "smelly,g,2026-10-01T12:00:00")
    assert unpushed_rows(local, base) == ["smelly,g,2026-10-01T12:00:00"]


def test_unpushed_rows_ignore_line_endings():
    base = _csv("a,g,1", crlf=False)
    local = _csv("a,g,1", "b,g,2", crlf=True)
    assert unpushed_rows(local, base) == ["b,g,2"]


def test_unpushed_rows_when_file_absent_upstream():
    assert unpushed_rows(_csv("a,g,1"), "") == ["a,g,1"]


def test_unpushed_rows_respects_duplicate_counts():
    base = _csv("a,g,1")
    local = _csv("a,g,1", "a,g,1")
    assert unpushed_rows(local, base) == ["a,g,1"]


def test_missing_rows_skips_rows_already_present():
    current = _csv("a,g,1", "b,g,2")
    assert missing_rows(current, ["b,g,2", "c,g,3"]) == ["c,g,3"]


def test_append_rows_adds_header_to_new_file(tmp_path):
    p = tmp_path / "adjectives.csv"
    append_rows(p, ["a,g,1"])
    assert p.read_bytes() == f"{HEADER}\r\na,g,1\r\n".encode()


def test_append_rows_fixes_missing_trailing_newline(tmp_path):
    p = tmp_path / "adjectives.csv"
    p.write_bytes(f"{HEADER}\r\na,g,1".encode())
    append_rows(p, ["b,g,2"])
    assert p.read_bytes() == f"{HEADER}\r\na,g,1\r\nb,g,2\r\n".encode()


# --- end to end against real git repos ----------------------------------------

def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def pi(tmp_path, monkeypatch):
    """A bare 'GitHub' remote, a dev clone, and a 'Pi' clone of main."""
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                 "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    dev = tmp_path / "dev"
    _git(tmp_path, "clone", "-q", str(remote), str(dev))
    _git(dev, "config", "core.autocrlf", "false")
    (dev / "insults" / "submissions").mkdir(parents=True)
    for f in SUBMISSION_FILES:
        (dev / f).write_bytes(_csv("seed,g,2026-05-01T00:00:00").encode())
    (dev / "code.py").write_text("v1\n")
    _git(dev, "add", "-A")
    _git(dev, "commit", "-q", "-m", "seed")
    _git(dev, "push", "-q", "origin", "main")
    pi = tmp_path / "pi"
    _git(tmp_path, "clone", "-q", "--single-branch", "-b", "main", str(remote), str(pi))
    _git(pi, "config", "core.autocrlf", "false")
    return {"remote": remote, "dev": dev, "pi": pi, "stash": tmp_path / "stash"}


def _remote_file(env, f):
    return subprocess.run(["git", "--git-dir", str(env["remote"]), "show", f"main:{f}"],
                          check=True, capture_output=True, text=True).stdout


def test_unpushed_submission_survives_boot_reset_and_is_pushed(pi):
    adj = SUBMISSION_FILES[0]
    # Kiosk records a submission and commits it, but the push never happens.
    with open(pi["pi"] / adj, "ab") as fh:
        fh.write(b"smelly,g,2026-10-27T10:00:00\r\n")
    _git(pi["pi"], "commit", "-q", "-am", "submission: g/smelly")
    # Meanwhile the maintainer pushes code to main.
    (pi["dev"] / "code.py").write_text("v2\n")
    _git(pi["dev"], "commit", "-q", "-am", "code v2")
    _git(pi["dev"], "push", "-q", "origin", "main")

    # --- boot: start.sh ---
    save(pi["pi"], pi["stash"])
    _git(pi["pi"], "fetch", "-q", "origin", "main")
    _git(pi["pi"], "reset", "-q", "--hard", "origin/main")
    assert "smelly" not in (pi["pi"] / adj).read_text()   # the reset wiped it
    restore(pi["pi"], pi["stash"])

    assert "smelly,g,2026-10-27T10:00:00" in (pi["pi"] / adj).read_text()
    assert "smelly,g,2026-10-27T10:00:00" in _remote_file(pi, adj)
    assert (pi["pi"] / "code.py").read_text() == "v2\n"   # code update kept
    assert not any(pi["stash"].glob("*.csv"))             # stash cleared


def test_uncommitted_submission_is_also_recovered(pi):
    noun = SUBMISSION_FILES[1]
    with open(pi["pi"] / noun, "ab") as fh:
        fh.write(b"goblin,old,2026-10-27T11:00:00\r\n")
    save(pi["pi"], pi["stash"])
    _git(pi["pi"], "reset", "-q", "--hard", "origin/main")
    restore(pi["pi"], pi["stash"])
    assert "goblin,old" in _remote_file(pi, noun)


def test_boot_with_nothing_unpushed_is_a_noop(pi):
    head = _git(pi["pi"], "rev-parse", "HEAD")
    save(pi["pi"], pi["stash"])
    restore(pi["pi"], pi["stash"])
    assert _git(pi["pi"], "rev-parse", "HEAD") == head


def test_failed_push_keeps_rows_for_next_boot(pi, monkeypatch):
    adj = SUBMISSION_FILES[0]
    with open(pi["pi"] / adj, "ab") as fh:
        fh.write(b"soggy,g,2026-10-27T12:00:00\r\n")
    save(pi["pi"], pi["stash"])
    _git(pi["pi"], "reset", "-q", "--hard", "origin/main")
    import submission_recovery
    monkeypatch.setattr(submission_recovery, "_run_sync", lambda **kw: False)
    restore(pi["pi"], pi["stash"])
    # Row is back on disk (committed locally), so the next boot's save finds it.
    assert "soggy" in (pi["pi"] / adj).read_text()
    save(pi["pi"], pi["stash"])
    assert "soggy" in "".join(p.read_text() for p in pi["stash"].glob("*.csv"))


def test_save_and_restore_never_raise_outside_a_repo(tmp_path):
    save(tmp_path / "not-a-repo", tmp_path / "stash")
    restore(tmp_path / "not-a-repo", tmp_path / "stash")
