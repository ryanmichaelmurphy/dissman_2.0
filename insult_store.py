"""CSV-backed storage for active insult words and user submissions.

Layout under base_dir:
    active/adjectives.csv         word,category,added_date
    active/nouns.csv              word,category,added_date
    submissions/adjectives.csv    word,category,submission_date
    submissions/nouns.csv         word,category,submission_date

Duplicates in the active CSVs are intentional and preserved on read so
that random.choice gives words drawn-with-replacement at a frequency
proportional to how often they appear.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

CATEGORIES = ("g", "r", "old")
PARTS_OF_SPEECH = ("adj", "noun")

CATEGORY_LABELS = {
    "g": "G-rated",
    "r": "R-rated",
    "old": "Old-timey",
}

_POS_FILE = {"adj": "adjectives.csv", "noun": "nouns.csv"}


class InsultStore:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.active_dir = self.base_dir / "active"
        self.submissions_dir = self.base_dir / "submissions"

    def _read_all(self, pos):
        path = self.active_dir / _POS_FILE[pos]
        if not path.exists():
            return []
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            return [
                (row["word"], row["category"])
                for row in reader
                if row.get("word") and row.get("category")
            ]

    def _filter(self, pos, category):
        rows = self._read_all(pos)
        if category == "all":
            return [w for w, _ in rows]
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        return [w for w, c in rows if c == category]

    def adjectives(self, category):
        return self._filter("adj", category)

    def nouns(self, category):
        return self._filter("noun", category)

    def record_submission(self, category, pos, word, now=None):
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        if pos not in PARTS_OF_SPEECH:
            raise ValueError(f"unknown pos: {pos}")
        clean = word.strip().lower()
        if not clean:
            raise ValueError("word cannot be empty")
        now = now or datetime.now()

        self.submissions_dir.mkdir(parents=True, exist_ok=True)
        path = self.submissions_dir / _POS_FILE[pos]
        new_file = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["word", "category", "submission_date"])
            writer.writerow([clean, category, now.isoformat(timespec="seconds")])
