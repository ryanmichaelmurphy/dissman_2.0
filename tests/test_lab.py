import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompt_testing"))

from lab import cache_is_fresh, load_row, key_from_systemd_unit


def test_cache_fresh_when_prompt_matches():
    assert cache_is_fresh({"prompt": "abc", "status": "accepted"}, "abc") is True


def test_cache_stale_when_prompt_changed():
    assert cache_is_fresh({"prompt": "old", "status": "accepted"}, "new") is False


def test_cache_stale_when_no_sidecar():
    assert cache_is_fresh(None, "abc") is False


def test_cache_stale_when_previously_rejected():
    # a rejected sidecar has no image, so it's never "fresh" for printing
    assert cache_is_fresh({"prompt": "abc", "status": "rejected"}, "abc") is False


def test_load_row_finds_by_name():
    rows = [{"name": "a"}, {"name": "b"}]
    assert load_row(rows, "b") == {"name": "b"}


def test_load_row_missing_returns_none():
    assert load_row([{"name": "a"}], "z") is None


def test_key_from_unit_plain():
    unit = "[Service]\nEnvironment=OPENAI_API_KEY=sk-abc123\nExecStart=/x\n"
    assert key_from_systemd_unit(unit) == "sk-abc123"


def test_key_from_unit_quoted():
    unit = 'Environment="OPENAI_API_KEY=sk-xyz"\n'
    assert key_from_systemd_unit(unit) == "sk-xyz"


def test_key_from_unit_absent_returns_none():
    assert key_from_systemd_unit("[Service]\nExecStart=/x\n") is None
