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


def test_run_sync_runs_add_commit_push_in_order(fake_run, tmp_path):
    _run_sync(repo_dir=tmp_path, message="add submission: g/adj/snazzy")
    args_list = [c.args[0] for c in fake_run.call_args_list]
    assert args_list[0][:2] == ["git", "add"]
    assert args_list[0][2] == "insults/submissions/"
    assert args_list[1][:2] == ["git", "commit"]
    assert args_list[2][:2] == ["git", "push"]


def test_run_sync_commit_message_used(fake_run, tmp_path):
    _run_sync(repo_dir=tmp_path, message="add submission: g/adj/snazzy")
    commit_call = fake_run.call_args_list[1].args[0]
    assert "add submission: g/adj/snazzy" in commit_call


def test_run_sync_swallows_subprocess_failures(tmp_path):
    with patch("github_sync.subprocess.run") as m:
        m.side_effect = subprocess.CalledProcessError(1, "git push")
        # should not raise
        _run_sync(repo_dir=tmp_path, message="m")


def test_push_submission_async_returns_immediately(tmp_path):
    with patch("github_sync._run_sync") as m:
        thread = push_submission_async(repo_dir=tmp_path, message="m")
        thread.join(timeout=2)
        m.assert_called_once_with(repo_dir=tmp_path, message="m")
