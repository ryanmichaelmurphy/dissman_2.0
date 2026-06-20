import json
from pathlib import Path

from prompt_store import PromptChoice
from print_pipeline import PrintSettings
from prompt_fallback import (
    build_candidates, read_last_successful, write_last_successful,
    FINAL_FALLBACK, FINAL_FALLBACK_NAME, Candidate,
)


def _choice(name, prompt, **kw):
    return PromptChoice(name, prompt, PrintSettings(**kw) if kw else PrintSettings.defaults())


FINAL = PromptChoice(FINAL_FALLBACK_NAME, FINAL_FALLBACK, PrintSettings.defaults())


def test_build_full_four_tier_order():
    cands = build_candidates(
        primary=_choice("p", "PRIMARY"),
        is_base=_choice("b", "BASE"),
        last_successful=_choice("l", "LAST"),
        final_fallback=FINAL,
    )
    assert [c.tier for c in cands] == ["primary", "is_base", "last_successful", "final_fallback"]
    assert [c.choice.prompt for c in cands] == ["PRIMARY", "BASE", "LAST", FINAL_FALLBACK]


def test_dedup_primary_equals_is_base_keeps_primary_settings():
    primary = _choice("p", "SAME", binarization="threshold", threshold=140)
    is_base = _choice("b", "SAME", binarization="dither", threshold=99)
    cands = build_candidates(primary, is_base, None, FINAL)
    tiers = [c.tier for c in cands]
    assert "is_base" not in tiers          # deduped
    primary_cand = next(c for c in cands if c.choice.prompt == "SAME")
    assert primary_cand.tier == "primary"
    assert primary_cand.choice.settings.threshold == 140   # earlier tier's settings win


def test_dedup_last_successful_equals_is_base():
    cands = build_candidates(
        primary=_choice("p", "PRIMARY"),
        is_base=_choice("b", "BASE"),
        last_successful=_choice("l", "BASE"),
        final_fallback=FINAL,
    )
    assert [c.tier for c in cands] == ["primary", "is_base", "final_fallback"]


def test_skips_none_and_empty_prompts():
    cands = build_candidates(
        primary=None,
        is_base=_choice("b", "   "),       # whitespace only
        last_successful=_choice("l", "LAST"),
        final_fallback=FINAL,
    )
    assert [c.tier for c in cands] == ["last_successful", "final_fallback"]


def test_final_fallback_dropped_when_duplicate():
    cands = build_candidates(
        primary=_choice("p", FINAL_FALLBACK),
        is_base=None, last_successful=None, final_fallback=FINAL,
    )
    assert [c.tier for c in cands] == ["primary"]   # final is a dup of primary text


def test_read_missing_file_returns_none(tmp_path):
    assert read_last_successful(tmp_path / "nope.json") is None


def test_read_corrupt_json_returns_none(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    assert read_last_successful(p) is None


def test_read_valid_returns_choice(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({
        "name": "n", "prompt": "P",
        "settings": {"binarization": "threshold", "threshold": 140, "contrast": 0.8,
                     "brightness": 1.0, "resize_width": 384, "sharpness": 1.0, "gamma": 1.0},
        "ts": 1,
    }))
    c = read_last_successful(p)
    assert c.name == "n" and c.prompt == "P"
    assert c.settings.binarization == "threshold"
    assert c.settings.threshold == 140


def test_read_garbage_settings_falls_back_to_defaults(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"name": "n", "prompt": "P", "settings": {"bogus": 1}}))
    c = read_last_successful(p)
    assert c is not None
    assert c.settings == PrintSettings.defaults()


def test_write_then_read_round_trip(tmp_path):
    p = tmp_path / "x.json"
    original = _choice("n", "P", binarization="threshold", threshold=140, contrast=0.8,
                       brightness=1.1, resize_width=384, sharpness=1.5, gamma=1.2)
    write_last_successful(p, original)
    back = read_last_successful(p)
    assert back.name == "n" and back.prompt == "P"
    assert back.settings == original.settings
    assert not (tmp_path / "x.json.tmp").exists()   # temp cleaned up


def test_write_overwrites_existing(tmp_path):
    p = tmp_path / "x.json"
    write_last_successful(p, _choice("a", "A"))
    write_last_successful(p, _choice("b", "B"))
    assert read_last_successful(p).name == "b"
