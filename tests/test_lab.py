import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompt_testing"))

from lab import (
    cache_is_fresh, cache_stem, estimate_cost, key_from_systemd_unit, load_row,
    usage_dict,
)


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

def test_cache_stale_when_quality_changed():
    sc = {"prompt": "abc", "status": "accepted", "quality": "high"}
    assert cache_is_fresh(sc, "abc", "low") is False
    assert cache_is_fresh(sc, "abc", "high") is True


def test_cache_stem_is_per_model():
    assert cache_stem("bully", "gpt-image-2.5-flare") == "bully__gpt-image-2.5-flare"
    assert cache_stem("bully", "a") != cache_stem("bully", "b")


def test_estimate_cost_uses_image_and_text_split():
    usage = {"input_tokens": 1100, "output_tokens": 200,
             "input_tokens_details": {"image_tokens": 1000, "text_tokens": 100}}
    # 1000*8 + 100*5 + 200*30 = 14500 per 1M
    assert estimate_cost(usage, "gpt-image-2.5-flare") == pytest.approx(0.0145)


def test_estimate_cost_mini_rates():
    usage = {"output_tokens": 1000,
             "input_tokens_details": {"image_tokens": 0, "text_tokens": 0}}
    assert estimate_cost(usage, "gpt-image-1-mini") == pytest.approx(0.008)


def test_estimate_cost_without_details_treats_input_as_image():
    usage = {"input_tokens": 1000, "output_tokens": 0}
    assert estimate_cost(usage, "unknown-model") == pytest.approx(0.008)


def test_estimate_cost_none_when_no_usage():
    assert estimate_cost(None, "gpt-image-2.5-flare") is None


def test_usage_dict_from_model_dump():
    class U:
        def model_dump(self):
            return {"output_tokens": 5}
    assert usage_dict(U()) == {"output_tokens": 5}
