import subprocess
from pathlib import Path
from unittest.mock import patch, call

import pytest

from github_sync import push_submission_async, _run_sync


@pytest.fixture
def fake_run():
    with patch("github_sync.subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        yield m


def _cmds(mock):
    return [c.args[0] for c in mock.call_args_list]


def test_run_sync_runs_add_commit_rebase_push_in_order(fake_run, tmp_path):
    _run_sync(repo_dir=tmp_path, message="add submission: g/adj/snazzy")
    cmds = _cmds(fake_run)
    assert cmds[0][:2] == ["git", "add"]
    assert cmds[0][2] == "insults/submissions/"
    assert cmds[1][:2] == ["git", "commit"]
    assert cmds[2][:3] == ["git", "pull", "--rebase"]
    assert cmds[3][:2] == ["git", "push"]
    assert len(cmds) == 4


def test_run_sync_commit_message_used(fake_run, tmp_path):
    _run_sync(repo_dir=tmp_path, message="add submission: g/adj/snazzy")
    commit_call = fake_run.call_args_list[1].args[0]
    assert "add submission: g/adj/snazzy" in commit_call


def test_rejected_push_rebases_and_retries_once(tmp_path):
    def side_effect(cmd, **kw):
        if cmd[:2] == ["git", "push"] and side_effect.pushes == 0:
            side_effect.pushes += 1
            raise subprocess.CalledProcessError(1, cmd, stderr=b"rejected (fetch first)")
        return subprocess.CompletedProcess(args=cmd, returncode=0)
    side_effect.pushes = 0

    with patch("github_sync.subprocess.run", side_effect=side_effect) as m:
        _run_sync(repo_dir=tmp_path, message="m")
    cmds = [c[:2] for c in _cmds(m)]
    assert cmds == [["git", "add"], ["git", "commit"], ["git", "pull"],
                    ["git", "push"], ["git", "pull"], ["git", "push"]]


def test_failed_rebase_is_aborted_and_not_pushed(tmp_path):
    def side_effect(cmd, **kw):
        if cmd[:2] == ["git", "pull"]:
            raise subprocess.CalledProcessError(1, cmd, stderr=b"conflict")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("github_sync.subprocess.run", side_effect=side_effect) as m:
        _run_sync(repo_dir=tmp_path, message="m")   # must not raise
    cmds = _cmds(m)
    assert ["git", "rebase", "--abort"] in cmds
    assert not any(c[:2] == ["git", "push"] for c in cmds)


def test_nothing_to_commit_still_pushes_earlier_commits(tmp_path):
    # A commit left behind by an earlier failed push must still go out.
    def side_effect(cmd, **kw):
        if cmd[:2] == ["git", "commit"]:
            raise subprocess.CalledProcessError(1, cmd, stderr=b"nothing to commit")
        return subprocess.CompletedProcess(args=cmd, returncode=0)

    with patch("github_sync.subprocess.run", side_effect=side_effect) as m:
        _run_sync(repo_dir=tmp_path, message="m")
    assert any(c[:2] == ["git", "push"] for c in _cmds(m))


def test_run_sync_swallows_subprocess_failures(tmp_path):
    with patch("github_sync.subprocess.run") as m:
        m.side_effect = subprocess.CalledProcessError(1, "git push")
        # should not raise
        _run_sync(repo_dir=tmp_path, message="m")


def test_run_sync_returns_success_flag(fake_run, tmp_path):
    assert _run_sync(repo_dir=tmp_path, message="m") is True
    with patch("github_sync.subprocess.run") as m:
        m.side_effect = subprocess.CalledProcessError(1, "git")
        assert _run_sync(repo_dir=tmp_path, message="m") is False


def test_push_submission_async_returns_immediately(tmp_path):
    with patch("github_sync._run_sync") as m:
        thread = push_submission_async(repo_dir=tmp_path, message="m")
        thread.join(timeout=2)
        m.assert_called_once_with(repo_dir=tmp_path, message="m")
