import csv
import random
from pathlib import Path

import pytest

from prompt_store import PromptStore, PromptChoice, FALLBACK_PROMPT
from print_pipeline import PrintSettings


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
    c = s.choose()
    assert c.name == "a"
    assert c.prompt == "the only prompt"


def test_choose_uses_weights(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "99", "is_base": "true", "prompt": "base prompt"},
        {"name": "rare", "weight": "1", "is_base": "false", "prompt": "rare prompt"},
    ])
    s = PromptStore(p)
    random.seed(0)
    picks = [s.choose().name for _ in range(1000)]
    base = picks.count("base")
    assert 950 < base < 999


def test_choose_falls_back_to_base_when_all_weights_zero(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "0", "is_base": "true", "prompt": "base prompt"},
        {"name": "alt", "weight": "0", "is_base": "false", "prompt": "alt prompt"},
    ])
    s = PromptStore(p)
    c = s.choose()
    assert c.name == "base"
    assert c.prompt == "base prompt"


def test_choose_falls_back_to_hardcoded_when_csv_missing(tmp_path):
    s = PromptStore(tmp_path / "nonexistent.csv")
    c = s.choose()
    assert c.name == "fallback"
    assert c.prompt == FALLBACK_PROMPT


def test_choose_falls_back_to_hardcoded_when_csv_empty(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [])
    s = PromptStore(p)
    assert s.choose().name == "fallback"


def test_negative_weights_treated_as_zero(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "10", "is_base": "true", "prompt": "base"},
        {"name": "broken", "weight": "-5", "is_base": "false", "prompt": "broken"},
    ])
    s = PromptStore(p)
    random.seed(42)
    for _ in range(100):
        assert s.choose().name == "base"


def test_non_numeric_weight_treated_as_zero(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    _write_csv(p, [
        {"name": "base", "weight": "10", "is_base": "true", "prompt": "base"},
        {"name": "broken", "weight": "oops", "is_base": "false", "prompt": "broken"},
    ])
    s = PromptStore(p)
    random.seed(1)
    for _ in range(100):
        assert s.choose().name == "base"


def test_choose_includes_print_settings(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    p.write_text(
        "name,weight,is_base,binarization,threshold,contrast,brightness,"
        "resize_width,sharpness,gamma,prompt\n"
        "a,1,true,threshold,140,0.8,1.1,384,1.5,1.2,a prompt\n",
        newline="",
    )
    c = PromptStore(p).choose()
    assert isinstance(c.settings, PrintSettings)
    assert c.settings.binarization == "threshold"
    assert c.settings.threshold == 140
    assert c.settings.contrast == 0.8


def test_fallback_choice_has_default_settings(tmp_path):
    c = PromptStore(tmp_path / "nope.csv").choose()
    assert c.settings == PrintSettings.defaults()


def test_base_returns_is_base_row_with_settings(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    p.write_text(
        "name,weight,is_base,binarization,threshold,contrast,brightness,"
        "resize_width,sharpness,gamma,prompt\n"
        "a,0,false,dither,128,0.3,1.0,380,1.0,1.0,not base\n"
        "b,1,true,threshold,150,0.9,1.0,384,1.0,1.0,the base\n",
        newline="",
    )
    c = PromptStore(p).base()
    assert c is not None
    assert c.name == "b"
    assert c.prompt == "the base"
    assert c.settings.binarization == "threshold"
    assert c.settings.threshold == 150


def test_base_returns_none_when_no_is_base(tmp_path):
    p = tmp_path / "drawing_prompts.csv"
    p.write_text(
        "name,weight,is_base,prompt\na,1,false,x\n", newline="",
    )
    assert PromptStore(p).base() is None


def test_base_tolerates_missing_csv(tmp_path):
    assert PromptStore(tmp_path / "nope.csv").base() is None
