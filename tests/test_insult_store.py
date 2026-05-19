import csv
from datetime import datetime
from pathlib import Path

import pytest

from insult_store import InsultStore, CATEGORIES, PARTS_OF_SPEECH


def _init_layout(base: Path) -> None:
    (base / "active").mkdir(parents=True)
    (base / "submissions").mkdir(parents=True)
    (base / "active" / "adjectives.csv").write_text("word,category,added_date\n")
    (base / "active" / "nouns.csv").write_text("word,category,added_date\n")
    (base / "submissions" / "adjectives.csv").write_text("word,category,submission_date\n")
    (base / "submissions" / "nouns.csv").write_text("word,category,submission_date\n")


@pytest.fixture
def store(tmp_path):
    base = tmp_path / "insults"
    _init_layout(base)
    return InsultStore(base_dir=base)


def test_load_returns_empty_lists_when_csvs_empty(store):
    assert store.adjectives("g") == []
    assert store.nouns("g") == []


def test_load_filters_by_category(tmp_path):
    base = tmp_path / "insults"
    _init_layout(base)
    (base / "active" / "adjectives.csv").write_text(
        "word,category,added_date\n"
        "clumsy,g,2026-05-18\n"
        "grumpy,g,2026-05-18\n"
        "shitty,r,2026-05-18\n"
        "froward,old,2026-05-18\n"
    )
    s = InsultStore(base_dir=base)
    assert s.adjectives("g") == ["clumsy", "grumpy"]
    assert s.adjectives("r") == ["shitty"]
    assert s.adjectives("old") == ["froward"]


def test_load_preserves_duplicates(tmp_path):
    """Duplicates increase draw probability - must NOT be deduplicated."""
    base = tmp_path / "insults"
    _init_layout(base)
    (base / "active" / "adjectives.csv").write_text(
        "word,category,added_date\n"
        "clumsy,g,2026-05-18\n"
        "clumsy,g,2026-05-18\n"
        "clumsy,g,2026-05-18\n"
        "grumpy,g,2026-05-18\n"
    )
    s = InsultStore(base_dir=base)
    assert s.adjectives("g") == ["clumsy", "clumsy", "clumsy", "grumpy"]


def test_anything_goes_unions_all_categories(tmp_path):
    base = tmp_path / "insults"
    _init_layout(base)
    (base / "active" / "adjectives.csv").write_text(
        "word,category,added_date\n"
        "clumsy,g,2026-05-18\n"
        "shitty,r,2026-05-18\n"
        "froward,old,2026-05-18\n"
    )
    s = InsultStore(base_dir=base)
    assert sorted(s.adjectives("all")) == ["clumsy", "froward", "shitty"]


def test_record_submission_appends_row(store):
    store.record_submission("g", "adj", "snazzy", now=datetime(2026, 5, 18, 12, 0, 0))
    with (store.base_dir / "submissions" / "adjectives.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [{
        "word": "snazzy", "category": "g",
        "submission_date": "2026-05-18T12:00:00",
    }]


def test_record_submission_writes_nouns_to_nouns_file(store):
    store.record_submission("r", "noun", "wombat", now=datetime(2026, 5, 18, 12, 0, 0))
    with (store.base_dir / "submissions" / "adjectives.csv").open() as f:
        adj_rows = list(csv.DictReader(f))
    with (store.base_dir / "submissions" / "nouns.csv").open() as f:
        noun_rows = list(csv.DictReader(f))
    assert adj_rows == []
    assert noun_rows[0]["word"] == "wombat"
    assert noun_rows[0]["category"] == "r"


def test_record_submission_normalizes_word(store):
    store.record_submission("g", "adj", "  Snazzy  ", now=datetime(2026, 5, 18, 12, 0, 0))
    with (store.base_dir / "submissions" / "adjectives.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["word"] == "snazzy"


def test_record_submission_rejects_bad_category(store):
    with pytest.raises(ValueError):
        store.record_submission("xxx", "adj", "word")


def test_record_submission_rejects_bad_pos(store):
    with pytest.raises(ValueError):
        store.record_submission("g", "verb", "word")


def test_record_submission_rejects_empty_word(store):
    with pytest.raises(ValueError):
        store.record_submission("g", "adj", "   ")


def test_categories_and_pos_constants():
    assert CATEGORIES == ("g", "r", "old")
    assert PARTS_OF_SPEECH == ("adj", "noun")
