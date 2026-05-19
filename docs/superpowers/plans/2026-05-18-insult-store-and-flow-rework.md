# Insult Store + User-Submitted Insults + Flow Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded insult dictionaries with versioned CSV files; add a "teach Dissman an insult" flow that captures a category/adjective/noun from the user and pushes the submission to GitHub; reorder screens so the API call runs in parallel with the user picking their insult.

**Architecture:**
- A new `insult_store.py` module owns loading active word lists from `insults/active/*.csv` and appending submissions to `insults/submissions.csv`. `dissman.py` calls into the store instead of holding word constants inline.
- A new `github_sync.py` module runs `git add/commit/push` in a background thread when a submission is recorded — offline-tolerant; failures log but don't block the kiosk flow.
- The screen flow is reordered: after category pick the user goes straight to the camera, the API call kicks off as soon as the photo is captured, and the insult-choice screen renders while the request is in flight. A short "thinking" screen only appears if the user picks before the image is ready.

**Tech Stack:** Python 3, Kivy (existing), `csv` stdlib, `subprocess` for git, `pytest` for unit tests on pure-logic modules.

---

## File Structure

**New files:**
- `insult_store.py` — load/append CSV word lists and submissions
- `github_sync.py` — async git commit+push for submissions
- `insults/active/adjectives.csv` — active adjectives (`word,category,added_date`)
- `insults/active/nouns.csv` — active nouns (`word,category,added_date`)
- `insults/submissions/adjectives.csv` — user-submitted adjectives (`word,category,submission_date`)
- `insults/submissions/nouns.csv` — user-submitted nouns (`word,category,submission_date`)
- `tests/test_insult_store.py` — unit tests for the store
- `tests/test_github_sync.py` — unit tests for the sync wrapper (subprocess-mocked)

**Modified files:**
- `dissman.py` — replace inline word constants; reorder screens; add TeachFlow screens
- `insultmaster3.kv` — add layouts for `TeachCategoryScreen`, `TeachAdjScreen`, `TeachNounScreen`; add "Insult Dissman" button to `DisplayScreen`

**Key contract:**
- Active CSV header: `word,category,added_date`. `category ∈ {g, r, old}`. `added_date` is `YYYY-MM-DD`.
- Submissions CSV header: `word,category,submission_date`. `submission_date` is ISO 8601 (`YYYY-MM-DDTHH:MM:SS`).
- **Duplicates are preserved on purpose.** A word appearing N times in the active CSV is N times more likely to be drawn. The store does NOT de-duplicate on read or on seed.

---

## Task 1: Build the insult_store module

**Files:**
- Create: `insult_store.py`
- Create: `tests/test_insult_store.py`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Write the failing tests**

`tests/test_insult_store.py`:

```python
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
    """Duplicates increase draw probability — must NOT be deduplicated."""
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
    rows = list(csv.DictReader((store.base_dir / "submissions" / "adjectives.csv").open()))
    assert rows == [{
        "word": "snazzy", "category": "g",
        "submission_date": "2026-05-18T12:00:00",
    }]


def test_record_submission_writes_nouns_to_nouns_file(store):
    store.record_submission("r", "noun", "wombat", now=datetime(2026, 5, 18, 12, 0, 0))
    adj_rows = list(csv.DictReader((store.base_dir / "submissions" / "adjectives.csv").open()))
    noun_rows = list(csv.DictReader((store.base_dir / "submissions" / "nouns.csv").open()))
    assert adj_rows == []
    assert noun_rows[0]["word"] == "wombat"
    assert noun_rows[0]["category"] == "r"


def test_record_submission_normalizes_word(store):
    store.record_submission("g", "adj", "  Snazzy  ", now=datetime(2026, 5, 18, 12, 0, 0))
    rows = list(csv.DictReader((store.base_dir / "submissions" / "adjectives.csv").open()))
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_insult_store.py -v`
Expected: `ModuleNotFoundError: No module named 'insult_store'`

- [ ] **Step 3: Implement `insult_store.py`**

```python
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
    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.active_dir = self.base_dir / "active"
        self.submissions_dir = self.base_dir / "submissions"

    def _read_all(self, pos: str) -> list[tuple[str, str]]:
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

    def _filter(self, pos: str, category: str) -> list[str]:
        rows = self._read_all(pos)
        if category == "all":
            return [w for w, _ in rows]
        if category not in CATEGORIES:
            raise ValueError(f"unknown category: {category}")
        return [w for w, c in rows if c == category]

    def adjectives(self, category: str) -> list[str]:
        return self._filter("adj", category)

    def nouns(self, category: str) -> list[str]:
        return self._filter("noun", category)

    def record_submission(
        self,
        category: str,
        pos: str,
        word: str,
        now: datetime | None = None,
    ) -> None:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_insult_store.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add insult_store.py tests/test_insult_store.py tests/__init__.py
git commit -m "Add InsultStore with CSV-backed active words and submissions log"
```

---

## Task 2: Build the github_sync module

**Files:**
- Create: `github_sync.py`
- Create: `tests/test_github_sync.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_github_sync.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_github_sync.py -v`
Expected: `ModuleNotFoundError: No module named 'github_sync'`

- [ ] **Step 3: Implement `github_sync.py`**

```python
"""Async git commit+push helper for user-submitted insults.

Failures are logged but never raised — the kiosk must not crash if the
network is down or git misconfigured.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path


def _run_sync(repo_dir: Path, message: str) -> None:
    try:
        subprocess.run(
            ["git", "add", "insults/submissions/"],
            cwd=repo_dir, check=True, timeout=15,
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_dir, check=True, timeout=15,
            capture_output=True,
        )
        subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=repo_dir, check=True, timeout=30,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        print(f"[github_sync] push failed: {e}")


def push_submission_async(repo_dir: Path, message: str) -> threading.Thread:
    t = threading.Thread(
        target=_run_sync, kwargs={"repo_dir": repo_dir, "message": message},
        daemon=True,
    )
    t.start()
    return t
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_github_sync.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add github_sync.py tests/test_github_sync.py
git commit -m "Add async github sync for submission pushes"
```

---

## Task 3: Seed the active CSVs from current word constants

**Files:**
- Create: `insults/active/adjectives.csv`
- Create: `insults/active/nouns.csv`
- Create: `insults/submissions/adjectives.csv` (header only)
- Create: `insults/submissions/nouns.csv` (header only)
- Create: `scripts/seed_active_csvs.py`

This task migrates the current `dissman.py:122-131` hardcoded strings into the new 2-file CSV layout. The current `g_adj` and `g_nouns` strings happen to contain the full list duplicated end-to-end — that duplication is preserved on purpose (it boosts G-rated word frequency in "Anything Goes"). Likewise `r_nouns` has `cumrag` and `cumslut` listed twice. Don't dedupe.

- [ ] **Step 1: Write `scripts/seed_active_csvs.py`**

```python
"""One-off: migrate inline word constants from dissman.py into CSVs.

Two output files: insults/active/adjectives.csv and insults/active/nouns.csv,
each with columns word,category,added_date. Duplicates are preserved so that
draws weighted by frequency work naturally.
"""
import csv
from pathlib import Path

g_adj = "clumsy scatterbrained grumpy sloppy cranky loony cheeky stubborn sneaky rascally mopey shifty snarky pouty grungy fussy sassy zonked knobby topsy-turvy clumsy scatterbrained grumpy sloppy cranky loony cheeky stubborn sneaky rascally mopey shifty snarky pouty grungy fussy sassy zonked knobby topsy-turvy"
g_nouns = "doodle hamburger backpack bedding bedspread binder blanket blinds bookcase book broom brush bucket calendar angler toad horse candle carpet chair china clock coffee-table comb comforter computer container couch credenza curtain cushion heater houseplant magnet mop radiator radio refrigerator rug saucer saw scissors screwdriver settee shade sheet shelf shirt shoe smoke-detector sneaker socks sofa speaker toy tool tv toothpaste towel nutcase toaster pancake muffin wombat caboose goblin pirate ninja meatball cupcake tadpole dingbat noodle turnip alien gadget grasshopper pickle wigwam bonnethead sharksucker"
r_adj = "shitty great-value pick-me horsefaced cum-guzzling christian vaginal straight cum semen smegma discharge fallopian anal aggressive arrogant boastful bossy boring careless clingy cruel cowardly deceitful dishonest greedy harsh impatient impulsive jealous moody narrow-minded overcritical rude selfish untrustworthy unhappy cumguzzling unfuckable incestuous sick perverted deranged depraved mountain-dew-drinking butterfaced self-centered revolting repellent repulsive sickening nauseating nauseous stomach-churning stomach-turning off-putting unpalatable unappetizing uninviting unsavoury distasteful foul nasty obnoxious odious"
r_nouns = "eunuch no-dick cum-for-brains toesniffer dicksucker dicksniffer toesucker simp skidmark shit-stain anal-fissure anal-wart anal-cyst vaginal-cyst vaginal-discharge smegma foreskin dick-cheese cumguzzler yuppy hippy karen boomer dork nerd dweeb unfuckable pedophile butterface needledick incel neckbeard wart genital-wart homewrecker doof douche doucheholster douchebag cum-receptacle cum-dumpster cumrag cumslut bum degenerate derelict good-for-nothing no-account no-good slacker hetrosexual buttlover breitbart-reader andriod-user trump-lover republican cumslut buttmuncher nutsack ballsack boner christian penis cunt twat asshole fucker shitbag shit-for-brains cumrag gland intestine cecum colon rectum liver gallbladder mesentery pancreas anus kidney ureter bladder urethra ovary tube uterus cervix discharge vagina"
old_adj = "froward pernickety laggardly moonstruck mumpsimus spleeny fribble dandiprat rattlecap slugabed cacafuego raggabrash dithering muddle-headed tatterdemalion claptrap wifty bedswerver lackadaisical flapdoodle"
old_nouns = "scallywag naysayer neerdowell landlover fustilarian snollygoster popinjay lickspittle rakefire whippersnapper noodle mumblecrust zounderkite gillywetfoot doodle pettifogger fopdoodle mooncalf clodpole hugger-mugger ragamuffin scalawag ninnyhammer flapdragon"

ADJ = [
    ("g", g_adj.split()),
    ("r", r_adj.split()),
    ("old", old_adj.split()),
]
NOUN = [
    ("g", g_nouns.split()),
    ("r", r_nouns.split()),
    ("old", old_nouns.split()),
]

ADDED = "2026-05-18"


def _write_active(path: Path, groups: list[tuple[str, list[str]]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["word", "category", "added_date"])
        for category, words in groups:
            for word in words:
                if word:
                    w.writerow([word, category, ADDED])
                    count += 1
    return count


def _write_submission_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow(["word", "category", "submission_date"])


def main():
    root = Path(__file__).resolve().parent.parent

    n_adj = _write_active(root / "insults" / "active" / "adjectives.csv", ADJ)
    print(f"wrote insults/active/adjectives.csv ({n_adj} rows)")

    n_noun = _write_active(root / "insults" / "active" / "nouns.csv", NOUN)
    print(f"wrote insults/active/nouns.csv ({n_noun} rows)")

    _write_submission_header(root / "insults" / "submissions" / "adjectives.csv")
    _write_submission_header(root / "insults" / "submissions" / "nouns.csv")
    print("wrote submission headers")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the seed script**

Run: `python scripts/seed_active_csvs.py`
Expected: 3 lines printed; row counts > 0.

- [ ] **Step 3: Spot-check the active adjectives CSV**

Run: `head -5 insults/active/adjectives.csv`
Expected:
```
word,category,added_date
clumsy,g,2026-05-18
scatterbrained,g,2026-05-18
grumpy,g,2026-05-18
sloppy,g,2026-05-18
```

- [ ] **Step 4: Verify duplicates are preserved**

Run: `grep -c "^clumsy,g," insults/active/adjectives.csv`
Expected: `2` (the original `g_adj` string contains the list twice).

- [ ] **Step 5: Verify InsultStore can read them**

Run:
```bash
python -c "from insult_store import InsultStore; s = InsultStore('insults'); print(len(s.adjectives('g')), len(s.nouns('g')), len(s.adjectives('all')))"
```
Expected: numbers > 0, with `all` equal to the sum of g+r+old counts.

- [ ] **Step 6: Commit**

```bash
git add insults/ scripts/seed_active_csvs.py
git commit -m "Seed active insult CSVs from existing word constants"
```

---

## Task 4: Wire dissman.py to load words from InsultStore

**Files:**
- Modify: `dissman.py:122-152` (remove inline word constants and `categories` dict)
- Modify: `dissman.py:173-187` (`InsultScreen.on_enter` and `generate_insults`)
- Modify: `dissman.py:443-447` (`CategoryScreen.select_category`)
- Modify: `dissman.py:540-581` (App init + populate_category_buttons)

- [ ] **Step 1: Remove the inline word constants and `categories` dict**

Delete `dissman.py:122-152` (everything from `# G-rated adjectives and nouns` through the closing `}` of the `categories` dict).

Replace with:

```python
from insult_store import InsultStore, CATEGORY_LABELS

INSULT_STORE = InsultStore(BASE_DIR / "insults")

# Category codes used internally: 'g', 'r', 'old', 'all'.
# Display labels come from CATEGORY_LABELS; 'all' is "Anything Goes".
CATEGORY_DISPLAY = [
    ("g", "G-rated"),
    ("r", "R-rated"),
    ("old", "Old-timey"),
    ("all", "Anything Goes"),
]
```

- [ ] **Step 2: Update `InsultScreen.on_enter` to use the store**

Replace `dissman.py:173-187`:

```python
class InsultScreen(Screen):
    def generate_insults(self, category: str) -> list[str]:
        adj_list = INSULT_STORE.adjectives(category)
        noun_list = INSULT_STORE.nouns(category)
        if not adj_list or not noun_list:
            return []
        return [
            f"{random.choice(adj_list)} {random.choice(noun_list)}"
            for _ in range(3)
        ]

    def on_enter(self, *args):
        self.ids.insult_options.clear_widgets()
        category = self.manager.current_category
        insults = self.generate_insults(category)
        for insult in insults:
            btn = ThemedButton(text=insult, size_hint_y=None, height=40)
            btn.bind(on_release=self.show_insult)
            self.ids.insult_options.add_widget(btn)
        self.ids.header.text = "What best describes you?"
        speak("Which insult best describes you?")
```

- [ ] **Step 3: Update `CategoryScreen` to store the category code (not display label)**

Replace `dissman.py:443-447`:

```python
class CategoryScreen(Screen):
    def select_category(self, category_code: str):
        self.manager.current_category = category_code
        self.manager.transition.direction = 'left'
        self.manager.current = 'insult'
```

- [ ] **Step 4: Update `populate_category_buttons` in `InsultMasterApp`**

Replace the body of `populate_category_buttons` (~`dissman.py:575-581`):

```python
def populate_category_buttons(self, *args):
    category_screen = self.sm.get_screen('category')
    for code, label in CATEGORY_DISPLAY:
        btn = ThemedButton(text=label, size_hint_y=None, height=40)
        btn.bind(on_release=lambda instance, c=code: category_screen.select_category(c))
        category_screen.ids.categories.add_widget(btn)
```

- [ ] **Step 5: Smoke-test the app launches on desktop**

Run: `python dissman.py`
Expected: app boots, splash plays, after the 5s simulated coin → category screen shows 4 buttons. Click each in turn; insult screen shows 3 generated options drawn from the CSVs (verify by picking "Old-timey" — adjectives should look like "froward/pernickety/etc.").

If the app crashes, fix before continuing. Close the window after smoke-testing.

- [ ] **Step 6: Commit**

```bash
git add dissman.py
git commit -m "Load insult words from InsultStore instead of inline constants"
```

---

## Task 5: Reorder flow so camera runs before insult choice and API runs in background

**Goal screen order:**

```
category → camera (capture, fires off API call in background) → insult (pick while API runs)
        → (if API done) display ; else → load (waits for API) → display
```

**Key idea:** the API call lives on the `ScreenManager` (or App), not on the `LoadScreen`. `CameraScreen` kicks it off, `LoadScreen` only handles the *waiting* animation and may be skipped entirely.

**Files:**
- Modify: `dissman.py:189-193` (`InsultScreen.show_insult`)
- Modify: `dissman.py:195-264` (`CameraScreen` — kick off API after capture)
- Modify: `dissman.py:269-325` (`LoadScreen` — read result from app state instead of fetching)
- Modify: `dissman.py:540-566` (`InsultMasterApp.build` — add state container)

- [ ] **Step 1: Add an `ImageJob` holder on the App**

In `InsultMasterApp.build`, after `self.sm = ScreenManager(...)`:

```python
self.image_job = ImageJob()
```

And add the class near the top of the module (after `speak()` and before the Kivy screen classes):

```python
class ImageJob:
    """Holds state for the in-flight or completed image generation."""

    def __init__(self):
        self.image_path: str | None = None
        self.ready: bool = False
        self.error: bool = False

    def reset(self):
        self.image_path = None
        self.ready = False
        self.error = False
```

- [ ] **Step 2: Move the API call out of `LoadScreen` into a standalone function**

Add to `dissman.py` (after `speak()`):

```python
def start_image_generation(source_image_path: str, job: "ImageJob", out_path: str) -> threading.Thread:
    """Fire the GPT image edit in a background thread. Updates `job` in place."""
    job.reset()

    def _work():
        try:
            with open(source_image_path, "rb") as f:
                response = client.images.edit(
                    model="gpt-image-1",
                    image=f,
                    prompt=(
                        "You are a middle school bully. Draw this person as a crude "
                        "middle school notebook doodle. Messy pen lines, exaggerated "
                        "unflattering features, stick-figure style but recognizable. "
                        "Make them uglier than they actually are with a stupid facial "
                        "expression."
                    ),
                    n=1,
                    size="1024x1024",
                )
            data = base64.b64decode(response.data[0].b64_json)
            with open(out_path, "wb") as f:
                f.write(data)
            job.image_path = out_path
            job.ready = True
        except Exception as e:
            print(f"[image-gen] failed: {e}")
            job.error = True

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    return t
```

- [ ] **Step 3: Update `CameraScreen.capture_image` to fire API call then go straight to insult screen**

Replace the body of `capture_image` (`dissman.py:243-264`):

```python
def capture_image(self, dt):
    ret, frame = self.camera.read()
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=50)
    if not ret:
        return

    timestamp = str(int(time.time()))
    save_path = path + 'test_' + timestamp + '.png'
    cv2.imwrite(save_path, frame)
    App.get_running_app().last_image_path = save_path
    self.ids.captured_image.source = save_path

    Clock.unschedule(self.update_preview)
    if self.camera.isOpened():
        self.camera.release()

    app = App.get_running_app()
    out_path = f'{path}downloaded_image_{timestamp}.png'
    start_image_generation(save_path, app.image_job, out_path)

    Clock.schedule_once(self.go_to_insult, 1.5)

def go_to_insult(self, dt):
    self.manager.transition.direction = 'left'
    self.manager.current = 'insult'
```

- [ ] **Step 4: Update `InsultScreen.show_insult` to skip `LoadScreen` if image is ready**

Replace `dissman.py:189-193`:

```python
def show_insult(self, instance):
    self.manager.get_screen('display').ids.insult_label.text = f"you {instance.text}."
    self.manager.transition.direction = 'left'
    app = App.get_running_app()
    if app.image_job.ready:
        self.manager.get_screen('display').ids.dall_e_image.source = app.image_job.image_path
        self.manager.current = 'display'
    else:
        self.manager.current = 'load'
```

- [ ] **Step 5: Update `LoadScreen` to wait on the in-flight job**

Replace `LoadScreen` (`dissman.py:269-325`):

```python
class LoadScreen(Screen):
    def on_enter(self, *args):
        if not hasattr(self, "image_widget"):
            self.image_widget = Image(source=f'{path}thinking0.png')
            self.add_widget(self.image_widget)
        self.current_image = 1
        speak("Thinking bad thoughts about you.")
        Clock.schedule_interval(self.check_image_ready, 0.3)

    def check_image_ready(self, dt):
        app = App.get_running_app()
        if app.image_job.error:
            self.manager.current = 'splash'
            return False

        num_images = 16
        self.current_image = (self.current_image + 1) % num_images
        self.image_widget.source = f'{path}thinking{self.current_image}.png'

        if app.image_job.ready:
            self.manager.get_screen('display').ids.dall_e_image.source = app.image_job.image_path
            self.manager.current = 'display'
            return False
```

- [ ] **Step 6: Smoke-test the new flow on desktop**

Run: `python dissman.py`. Insert simulated coin → pick category → camera screen appears, capture takes ~1.5s → insult screen renders, three options visible. Pick one quickly: if image isn't ready, the thinking animation should appear and then jump to display. If you wait longer before picking, you should go straight to display.

(Note: real `gpt-image-1` calls will be made and will cost you credits during smoke test. Set a fake/cheap key if you want to test flow without the cost — the error path should land on splash.)

- [ ] **Step 7: Commit**

```bash
git add dissman.py
git commit -m "Reorder flow: kick off image gen during/after camera, choose insult while it runs"
```

---

## Task 6: Add the "Insult Dissman" button to DisplayScreen

**Files:**
- Modify: `insultmaster3.kv` (DisplayScreen — add a button container above the QR button)
- Modify: `dissman.py:327-365` (`DisplayScreen.on_enter` — wire button)

- [ ] **Step 1: Update `insultmaster3.kv` — add `teach_button` block to DisplayScreen**

In the `<DisplayScreen>` block, immediately above the existing `BoxLayout: id: qr_button`, add:

```kv
        BoxLayout:
            id: teach_button
            orientation: 'vertical'
            padding: app.theme_vars['padding']
            spacing: app.theme_vars['spacing']
            size_hint_y: None
            height: 40
```

- [ ] **Step 2: Wire the button in `DisplayScreen.on_enter`**

After `self.ids.qr_button.clear_widgets()` (around `dissman.py:336`):

```python
        self.ids.teach_button.clear_widgets()
        teach_btn = ThemedButton(
            text="Insult Dissman to teach him new insults",
            size_hint_y=None, height=40,
        )
        teach_btn.bind(on_release=lambda x: self.go_to_teach())
        self.ids.teach_button.add_widget(teach_btn)
```

And add the handler to the class:

```python
def go_to_teach(self):
    # Cancel the auto-return-to-splash timer; teach flow owns the return.
    if hasattr(self, "_return_event") and self._return_event:
        self._return_event.cancel()
    self.has_entered = False
    self.manager.transition.direction = 'left'
    self.manager.current = 'teach_category'
```

Also: change the existing `Clock.schedule_once(lambda x: self.cleanup_and_restart(), 10)` to assign to `self._return_event`:

```python
self._return_event = Clock.schedule_once(lambda x: self.cleanup_and_restart(), 10)
```

- [ ] **Step 3: Smoke-test (skip — wired in the next task once the target screen exists)**

The button will crash with "No screen named 'teach_category'" until Task 7. That's fine; smoke-test happens at the end of Task 8.

- [ ] **Step 4: Commit**

```bash
git add dissman.py insultmaster3.kv
git commit -m "Add 'Insult Dissman' button on DisplayScreen"
```

---

## Task 7: Build the TeachCategoryScreen

**Files:**
- Modify: `insultmaster3.kv` — add `<TeachCategoryScreen>` block
- Modify: `dissman.py` — add `TeachCategoryScreen` class and register

- [ ] **Step 1: Add the kv block**

Append to `insultmaster3.kv`:

```kv
<TeachCategoryScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: app.theme_vars['padding']
        spacing: app.theme_vars['spacing']
        canvas.before:
            Color:
                rgba: app.theme_colors['background']
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: 'Teach Dissman an insult'
            font_name: app.fonts['heading']
            font_size: '24sp'
            color: app.theme_colors['primary']
            size_hint_y: None
            height: 40
        Label:
            text: 'Which category?'
            font_name: app.fonts['body']
            font_size: '18sp'
            color: app.theme_colors['secondary']
            size_hint_y: None
            height: 30
        BoxLayout:
            id: teach_categories
            orientation: 'vertical'
            size_hint_y: None
            height: self.minimum_height
            padding: app.theme_vars['padding']
            spacing: app.theme_vars['spacing']
```

- [ ] **Step 2: Add the screen class to `dissman.py`**

Above `class SplashScreen`:

```python
class TeachCategoryScreen(Screen):
    def on_enter(self, *args):
        self.ids.teach_categories.clear_widgets()
        speak("Pick a category for your insult.")
        for code, label in [("g", "G-rated"), ("r", "R-rated"), ("old", "Old-timey")]:
            btn = ThemedButton(text=label, size_hint_y=None, height=50)
            btn.bind(on_release=lambda inst, c=code: self.select(c))
            self.ids.teach_categories.add_widget(btn)

    def select(self, category_code: str):
        App.get_running_app().teach_submission = {"category": category_code}
        self.manager.transition.direction = 'left'
        self.manager.current = 'teach_adj'
```

- [ ] **Step 3: Register the screen in `InsultMasterApp.build`**

After `self.sm.add_widget(DisplayScreen(name='display'))`:

```python
self.sm.add_widget(TeachCategoryScreen(name='teach_category'))
```

And initialize state in `build`:

```python
self.teach_submission = {}
```

- [ ] **Step 4: Commit**

```bash
git add dissman.py insultmaster3.kv
git commit -m "Add TeachCategoryScreen for user-submitted insults"
```

---

## Task 8: Build TeachAdjScreen and TeachNounScreen with on-screen keyboard

Kivy ships a virtual keyboard (`VKeyboard`) and `TextInput` requests one automatically when `Window.softinput_mode` is set. For a fullscreen kiosk we want the keyboard visible without relying on system softinput, so we use `TextInput` with `keyboard_mode='managed'` and explicitly request a `VKeyboard` widget docked at the bottom.

**Files:**
- Modify: `dissman.py` — add `TeachAdjScreen`, `TeachNounScreen`, shared keyboard widget builder
- Modify: `insultmaster3.kv` — add `<TeachWordScreen>` block reused by both
- Modify: `dissman.py:540-566` — register new screens

- [ ] **Step 1: Add the kv block**

Append to `insultmaster3.kv`:

```kv
<TeachWordScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: app.theme_vars['padding']
        spacing: app.theme_vars['spacing']
        canvas.before:
            Color:
                rgba: app.theme_colors['background']
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            id: prompt_label
            text: ''
            font_name: app.fonts['heading']
            font_size: '22sp'
            color: app.theme_colors['primary']
            size_hint_y: None
            height: 40
        TextInput:
            id: word_input
            multiline: False
            font_name: app.fonts['body']
            font_size: '24sp'
            size_hint_y: None
            height: 50
            keyboard_mode: 'managed'
        BoxLayout:
            id: keyboard_holder
            size_hint_y: 1
        BoxLayout:
            id: action_row
            orientation: 'horizontal'
            size_hint_y: None
            height: 50
            spacing: 10
```

- [ ] **Step 2: Add a `TeachWordScreen` base class and the two concrete subclasses**

In `dissman.py` (above `class SplashScreen`):

```python
from kivy.uix.vkeyboard import VKeyboard


class TeachWordScreen(Screen):
    prompt = ""
    next_screen = ""
    pos_key = ""

    def on_enter(self, *args):
        self.ids.prompt_label.text = self.prompt
        self.ids.word_input.text = ""
        speak(self.prompt)

        # Build/replace the on-screen keyboard
        self.ids.keyboard_holder.clear_widgets()
        kb = VKeyboard(layout='qwerty', size_hint=(1, 1))
        kb.bind(on_key_up=self._on_key)
        self.ids.keyboard_holder.add_widget(kb)

        # Action buttons
        self.ids.action_row.clear_widgets()
        cancel = ThemedButton(text="Cancel")
        cancel.bind(on_release=lambda x: self.cancel())
        submit = ThemedButton(text="Submit")
        submit.bind(on_release=lambda x: self.submit())
        self.ids.action_row.add_widget(cancel)
        self.ids.action_row.add_widget(submit)

    def _on_key(self, keyboard, key, *args):
        # `key` is a 4-tuple (display, key_code, special, ascii_code) per Kivy docs.
        display, key_code, special, ascii_code = key
        current = self.ids.word_input.text
        if special == 'backspace':
            self.ids.word_input.text = current[:-1]
        elif special == 'enter':
            self.submit()
        elif special == 'spacebar':
            self.ids.word_input.text = current + ' '
        elif display and len(display) == 1:
            self.ids.word_input.text = current + display

    def submit(self):
        word = self.ids.word_input.text.strip().lower()
        if not word:
            speak("Type something first.")
            return
        app = App.get_running_app()
        app.teach_submission[self.pos_key] = word
        self.manager.transition.direction = 'left'
        self.manager.current = self.next_screen

    def cancel(self):
        App.get_running_app().teach_submission = {}
        self.manager.transition.direction = 'right'
        self.manager.current = 'splash'


class TeachAdjScreen(TeachWordScreen):
    prompt = "Type the adjective"
    pos_key = "adj"
    next_screen = "teach_noun"


class TeachNounScreen(TeachWordScreen):
    prompt = "Type the noun"
    pos_key = "noun"
    next_screen = "teach_submit"
```

- [ ] **Step 3: Register the screens**

In `InsultMasterApp.build`, after the `TeachCategoryScreen` registration:

```python
self.sm.add_widget(TeachAdjScreen(name='teach_adj'))
self.sm.add_widget(TeachNounScreen(name='teach_noun'))
```

Also register `<TeachAdjScreen>` and `<TeachNounScreen>` to inherit the kv rule by appending to `insultmaster3.kv`:

```kv
<TeachAdjScreen@TeachWordScreen>:
<TeachNounScreen@TeachWordScreen>:
```

(If Kivy's class-name matching already picks up the inherited rule without this, the lines are harmless; keep them for explicitness.)

- [ ] **Step 4: Smoke-test the two keyboard screens**

Run: `python dissman.py`. Drive: coin → category → camera → insult → display → tap "Insult Dissman" → pick a category → type an adjective with the on-screen keyboard → Submit → type a noun → Submit (transition to `teach_submit` will fail until Task 9; that's expected). Confirm the keyboard takes input including backspace and spacebar.

- [ ] **Step 5: Commit**

```bash
git add dissman.py insultmaster3.kv
git commit -m "Add TeachAdjScreen and TeachNounScreen with on-screen keyboard"
```

---

## Task 9: Submit screen + record + async git push

**Files:**
- Modify: `dissman.py` — add `TeachSubmitScreen` that calls `InsultStore.record_submission` and fires `github_sync.push_submission_async`

- [ ] **Step 1: Add the screen class**

In `dissman.py` (above `class SplashScreen`):

```python
import github_sync


class TeachSubmitScreen(Screen):
    def on_enter(self, *args):
        app = App.get_running_app()
        sub = app.teach_submission
        category = sub.get("category")
        adj = sub.get("adj")
        noun = sub.get("noun")
        app.teach_submission = {}

        try:
            INSULT_STORE.record_submission(category, "adj", adj)
            INSULT_STORE.record_submission(category, "noun", noun)
        except ValueError as e:
            print(f"[teach] rejected: {e}")
            speak("That didn't work. Try again later.")
            Clock.schedule_once(lambda dt: self._go_home(), 2)
            return

        msg = f"submission: {category}/{adj} {noun}"
        github_sync.push_submission_async(repo_dir=BASE_DIR, message=msg)

        speak(f"Thanks. I will remember: {adj} {noun}.")
        Clock.schedule_once(lambda dt: self._go_home(), 3)

    def _go_home(self):
        self.manager.transition.direction = 'right'
        self.manager.current = 'splash'
```

- [ ] **Step 2: Register the screen**

In `InsultMasterApp.build`, after `TeachNounScreen`:

```python
self.sm.add_widget(TeachSubmitScreen(name='teach_submit'))
```

- [ ] **Step 3: Optionally add a minimal kv block (just background)**

Append to `insultmaster3.kv`:

```kv
<TeachSubmitScreen>:
    BoxLayout:
        orientation: 'vertical'
        canvas.before:
            Color:
                rgba: app.theme_colors['background']
            Rectangle:
                size: self.size
                pos: self.pos
        Label:
            text: 'Got it.'
            font_name: app.fonts['heading']
            font_size: '32sp'
            color: app.theme_colors['primary']
```

- [ ] **Step 4: Smoke-test the full submission flow**

Run: `python dissman.py`. Drive: coin → ... → display → "Insult Dissman" → pick "G-rated" → type "snazzy" → Submit → type "wombat" → Submit → confirmation flashes, returns to splash. Then:

Run: `tail -1 insults/submissions/adjectives.csv && tail -1 insults/submissions/nouns.csv`
Expected: a new row `snazzy,g,...` in the adjectives file and `wombat,g,...` in the nouns file.

Run: `git log -1 --oneline`
Expected: a new "submission: g/snazzy wombat" commit (if push succeeded — if you're offline or the push fails, the commit may still exist locally; either is acceptable behaviour).

- [ ] **Step 5: Commit**

```bash
git add dissman.py insultmaster3.kv
git commit -m "Add TeachSubmitScreen: record submission and async-push to GitHub"
```

---

## Task 10: Update CLAUDE.md and deploy

**Files:**
- Modify: `CLAUDE.md` — document the new file layout, submission flow, screen order
- Modify: `dissman.py:122-152` cross-reference (already removed) — sanity check

- [ ] **Step 1: Update CLAUDE.md**

In `CLAUDE.md`, replace the screen-flow diagram and the "Architecture" section to reflect:

- New flow: `SplashScreen → CategoryScreen → CameraScreen → InsultScreen → (LoadScreen if needed) → DisplayScreen → [TeachCategoryScreen → TeachAdjScreen → TeachNounScreen → TeachSubmitScreen] → Splash`
- Word lists now live in `insults/active/{adjectives,nouns}.csv` with columns `word,category,added_date`
- User submissions are appended to `insults/submissions/{adjectives,nouns}.csv` and pushed to GitHub asynchronously via `github_sync.py`
- Duplicates in active CSVs are intentional — a word appearing N times is drawn N times more often
- To "approve" a submission: copy the row from `insults/submissions/*.csv` into `insults/active/*.csv` (add it twice or three times for more frequent use); the Pi will pull on next boot via `start.sh`

- [ ] **Step 2: Final desktop smoke test**

Run: `python dissman.py` end-to-end once more. Confirm: insult-pick happens while API is running; display shows the doodle; "Insult Dissman" button works; submission lands in `insults/submissions.csv`.

- [ ] **Step 3: Commit and push**

```bash
git add CLAUDE.md
git commit -m "Document insult-store / teach-flow / reordered screens"
git push origin main
```

- [ ] **Step 4: Deploy to the Pi**

SSH to the Pi (or use the MCP exec tool) and:

```bash
sudo systemctl restart dissman.service
sleep 12
tail /home/dissman/Documents/app/boot-update.log
sudo journalctl -u dissman.service -b --no-pager | tail -30
```

Expected: boot-update.log shows the new SHA, service is `active (running)`, no python tracebacks in journal.

- [ ] **Step 5: Live kiosk test**

Insert a coin (or wait for the simulated trigger if testing on bench), drive through the whole flow including teaching one insult. Verify it appears in `insults/submissions.csv` on GitHub.

---

## Notes for the implementer

- **Don't** restore the `# insult_text format is "you [adj] [noun]."` parsing in `DisplayScreen.on_enter` — it splits on whitespace and breaks for hyphenated nouns like "coffee-table". This plan doesn't change that behavior; just don't add similar parsing.
- The Pi has the GitHub remote at `https://github.com/ryanmichaelmurphy/dissman_2.0.git` with credentials already configured for push. If you change repos, update `start.sh` accordingly.
- The submissions CSV is the source of truth for *proposed* words. The active CSVs are curated by the owner. Don't auto-promote.
- Don't introduce a database. Plain CSVs are the design intent — they diff cleanly in git and are easy to hand-edit during review.
