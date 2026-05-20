import csv
import random
from pathlib import Path

import pytest

from prompt_store import PromptStore, FALLBACK_PROMPT


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "weight", "is_base", "prompt"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_choose_returns_prompt_text(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "a", "weight": "1", "is_base": "true", "prompt": "the only prompt"},
    ])
    s = PromptStore(p)
    name, prompt = s.choose()
    assert name == "a"
    assert prompt == "the only prompt"


def test_choose_uses_weights(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "99", "is_base": "true", "prompt": "base prompt"},
        {"name": "rare", "weight": "1", "is_base": "false", "prompt": "rare prompt"},
    ])
    s = PromptStore(p)
    random.seed(0)
    picks = [s.choose()[0] for _ in range(1000)]
    base = picks.count("base")
    assert 950 < base < 999


def test_choose_falls_back_to_base_when_all_weights_zero(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "0", "is_base": "true", "prompt": "base prompt"},
        {"name": "alt", "weight": "0", "is_base": "false", "prompt": "alt prompt"},
    ])
    s = PromptStore(p)
    name, prompt = s.choose()
    assert name == "base"
    assert prompt == "base prompt"


def test_choose_falls_back_to_hardcoded_when_csv_missing(tmp_path):
    s = PromptStore(tmp_path / "nonexistent.csv")
    name, prompt = s.choose()
    assert name == "fallback"
    assert prompt == FALLBACK_PROMPT


def test_choose_falls_back_to_hardcoded_when_csv_empty(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [])
    s = PromptStore(p)
    name, prompt = s.choose()
    assert name == "fallback"


def test_negative_weights_treated_as_zero(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "10", "is_base": "true", "prompt": "base"},
        {"name": "broken", "weight": "-5", "is_base": "false", "prompt": "broken"},
    ])
    s = PromptStore(p)
    random.seed(42)
    for _ in range(100):
        name, _ = s.choose()
        assert name == "base"


def test_non_numeric_weight_treated_as_zero(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "10", "is_base": "true", "prompt": "base"},
        {"name": "broken", "weight": "oops", "is_base": "false", "prompt": "broken"},
    ])
    s = PromptStore(p)
    random.seed(1)
    for _ in range(100):
        name, _ = s.choose()
        assert name == "base"
