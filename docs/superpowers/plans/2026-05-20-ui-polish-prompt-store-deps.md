# UI Polish, Prompt Store, and Dependency Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three independent improvements bundled for one deploy cycle — (1) small UI/asset/wording polish across the kiosk, (2) externalize the GPT-image prompt into a weighted CSV (mirroring the insult-word CSV pattern), (3) pin every dependency version so future installs don't drift.

**Architecture:**
- **UI polish** is pure Kivy/asset edits (insultmaster3.kv + dissman.py).
- **Prompt store** is a new `prompt_store.py` module with a `PromptStore.choose()` method that reads `prompts/drawing_prompts.csv` (columns: `name,weight,is_base,prompt`) and returns a prompt via weighted random selection. `start_image_generation` calls `PromptStore.choose()` at call time so each run can pick a different prompt.
- **Dependency pinning** is a single `requirements.txt` at the repo root with `==` pins for every direct dependency, plus a short note in `CLAUDE.md` describing how to install/regenerate it.

**Tech Stack:** Python 3.12 (dev env), Kivy 2.3.1, `csv` stdlib, `random.choices` for weighted draws, `pytest` for the prompt-store unit tests.

---

## File Structure

**New:**
- `prompt_store.py` — `PromptStore` class wrapping the CSV.
- `prompts/drawing_prompts.csv` — `name,weight,is_base,prompt` rows.
- `tests/test_prompt_store.py` — unit tests.
- `requirements.txt` — pinned versions.

**Modified:**
- `dissman.py`:
  - `start_image_generation` switches from a hardcoded prompt to `PROMPT_STORE.choose()`.
  - `CameraScreen` gains a capture-flash animation and an explicit slide transition to `InsultScreen`.
  - `TeachCategoryScreen` gets a `face.png` Image added in kv.
  - `TeachAdjScreen.prompt` / `TeachNounScreen.prompt` strings updated.
  - `TeachSubmitScreen` delay extended.
- `insultmaster3.kv`:
  - `<CategoryScreen>` `face.png` Image height updated.
  - `<TeachCategoryScreen>` gains a `face.png` Image block.
- `CLAUDE.md` — adds a "Dependencies" section pointing at `requirements.txt` plus the dev/Pi install commands.

**Open clarification (handled inline by Task 4):** The "shifty <something>" wording change is not pinned to a specific source line yet; Task 4 starts with a confirmation step. Everything else has concrete targets.

---

## Task 1: Build the PromptStore module + tests

**Files:**
- Create: `prompt_store.py`
- Create: `tests/test_prompt_store.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_prompt_store.py`:

```python
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
    # Expect ~990 base, ~10 rare; loose bounds for seeded randomness.
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prompt_store.py -v`
Expected: collection error / `ModuleNotFoundError: No module named 'prompt_store'`.

- [ ] **Step 3: Implement `prompt_store.py`**

```python
"""CSV-backed store for GPT-image drawing prompts.

CSV columns: name,weight,is_base,prompt

Selection is weighted-random over the rows whose weight parses to a positive
float. Non-numeric, negative, or zero weights are excluded from the draw.

If all weights are zero/invalid, the row marked is_base=true is returned. If
the CSV is missing or empty, FALLBACK_PROMPT is returned under the name
'fallback'.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


FALLBACK_PROMPT = (
    "You are a middle school bully. Draw this person as a crude middle school "
    "notebook doodle. Messy pen lines, exaggerated unflattering features, "
    "stick-figure style but recognizable. Make them uglier than they actually "
    "are with a stupid facial expression."
)


def _parse_weight(raw):
    try:
        w = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return w if w > 0 else 0.0


def _is_truthy(raw):
    return str(raw).strip().lower() in ("true", "1", "yes")


class PromptStore:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def _load(self):
        if not self.csv_path.exists():
            return []
        with self.csv_path.open(newline="") as f:
            return list(csv.DictReader(f))

    def choose(self):
        rows = self._load()
        if not rows:
            return ("fallback", FALLBACK_PROMPT)

        weighted = [(r, _parse_weight(r.get("weight"))) for r in rows]
        positive = [(r, w) for r, w in weighted if w > 0]

        if positive:
            choice = random.choices(
                [r for r, _ in positive],
                weights=[w for _, w in positive],
                k=1,
            )[0]
            return (choice.get("name", ""), choice.get("prompt", ""))

        for r in rows:
            if _is_truthy(r.get("is_base")):
                return (r.get("name", ""), r.get("prompt", ""))

        return ("fallback", FALLBACK_PROMPT)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompt_store.py -v`
Expected: **7 passed**.

- [ ] **Step 5: Commit**

```bash
git add prompt_store.py tests/test_prompt_store.py
git commit -m "Add PromptStore: CSV-backed weighted prompt selection"
```

---

## Task 2: Seed `prompts/drawing_prompts.csv`

**Files:**
- Create: `prompts/drawing_prompts.csv`

- [ ] **Step 1: Write the CSV**

Create `prompts/drawing_prompts.csv`. Use proper CSV quoting because the prompts contain commas:

```csv
name,weight,is_base,prompt
middle_school_bully,9,true,"You are a middle school bully. Draw this person as a crude middle school notebook doodle. Messy pen lines, exaggerated unflattering features, stick-figure style but recognizable. Make them uglier than they actually are with a stupid facial expression."
goya_saturn,1,false,"Draw the person in the image as a hideous old person in the style of Francisco de Goya's Black Paintings, in particular the painting Saturn Devouring His Son."
```

(Weights 9 and 1 give the requested ~1-in-10 rate for the Goya variant. To adjust, edit the weights — no code change.)

- [ ] **Step 2: Verify the store reads it**

```bash
python -c "from prompt_store import PromptStore; s = PromptStore('prompts/drawing_prompts.csv'); print(s.choose())"
```

Expected: prints a tuple, almost always `('middle_school_bully', '...')`. Run a few times to see the occasional `goya_saturn`.

- [ ] **Step 3: Commit**

```bash
git add prompts/drawing_prompts.csv
git commit -m "Seed drawing prompts CSV with bully (base) and Goya variants"
```

---

## Task 3: Wire `start_image_generation` to use PromptStore

**Files:**
- Modify: `dissman.py` — replace inline prompt in `start_image_generation` (lines ~132-163) with a call to a module-level `PROMPT_STORE.choose()`.

- [ ] **Step 1: Add PromptStore instance near `INSULT_STORE`**

Find the existing `INSULT_STORE = InsultStore(BASE_DIR / "insults")` line in `dissman.py`. Immediately after it, add:

```python
from prompt_store import PromptStore
PROMPT_STORE = PromptStore(BASE_DIR / "prompts" / "drawing_prompts.csv")
```

- [ ] **Step 2: Replace the hardcoded prompt in `start_image_generation`**

In `start_image_generation`, replace the `prompt=(...multi-line string...)` argument to `client.images.edit(...)` with `prompt=prompt_text` where `prompt_text` is chosen at the start of `_work()`:

```python
    def _work():
        prompt_name, prompt_text = PROMPT_STORE.choose()
        print(f"[image-gen] using prompt '{prompt_name}'")
        try:
            with open(source_image_path, "rb") as f:
                response = client.images.edit(
                    model="gpt-image-1",
                    image=f,
                    prompt=prompt_text,
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
```

- [ ] **Step 3: Verify**

Run:
```bash
python -c "import ast; ast.parse(open('dissman.py').read())"
python -m pytest tests/ -q
```

Expected: 22 passed (15 existing + 7 from Task 1).

- [ ] **Step 4: Commit**

```bash
git add dissman.py
git commit -m "Load drawing prompt from PromptStore (CSV) per call"
```

---

## Task 4: (DROPPED — wording stays as "This is what you look like..." per user)

No-op. Revisit later if user wants to change it.

---

## Task 5: Resize Dissman face on CategoryScreen

**Files:**
- Modify: `insultmaster3.kv` (CategoryScreen block, around line 21-24)

The current Image is `height: 165` — bumping it makes Dissman more prominent on the choice screen. Pick `220` as the new default (still leaves comfortable room for the title, header, and 4 category buttons on the 480-tall display).

- [ ] **Step 1: Edit**

In `insultmaster3.kv`, find the `<CategoryScreen>` block. Change:

```kv
        Image:
            source: 'face.png'
            size_hint_y: None
            height: 165
```

to:

```kv
        Image:
            source: 'face.png'
            size_hint_y: None
            height: 220
```

- [ ] **Step 2: Smoke-test the layout**

Run `python dissman.py` and confirm the four category buttons are still fully visible below the image. If they're pushed off-screen, reduce to 200 or 180.

- [ ] **Step 3: Commit**

```bash
git add insultmaster3.kv
git commit -m "Enlarge Dissman face on CategoryScreen (165 -> 220)"
```

---

## Task 6: Add capture flash + slide transition to InsultScreen

**Files:**
- Modify: `dissman.py` (CameraScreen.capture_image and on_enter; ScreenManager transition setup)

**Design:**
- **Flash (intra-screen, between preview and capture):** when `capture_image` fires, overlay a full-screen `Widget` with a white `Rectangle` and animate its opacity from 1.0 to 0.0 over 0.3s. This is the "snapped a photo" feel.
- **Slide (camera → insult):** swap `NoTransition` to a `SlideTransition` for the specific `go_to_insult` step, then restore `NoTransition` after, so other screens still cut.

- [ ] **Step 1: Add the imports**

Near other kivy imports in `dissman.py`:

```python
from kivy.uix.widget import Widget
from kivy.animation import Animation
from kivy.uix.screenmanager import SlideTransition
```

(Keep `NoTransition` as the default.)

- [ ] **Step 2: Add a `_flash_overlay` widget setup helper to `CameraScreen`**

Add to `CameraScreen` (anywhere in the class):

```python
    def _flash(self):
        from kivy.graphics import Color, Rectangle
        if not hasattr(self, "flash_widget"):
            self.flash_widget = Widget(size_hint=(1, 1), opacity=0)
            with self.flash_widget.canvas:
                Color(1, 1, 1, 1)
                self.flash_rect = Rectangle(pos=self.pos, size=self.size)
            self.flash_widget.bind(
                pos=lambda inst, val: setattr(self.flash_rect, "pos", val),
                size=lambda inst, val: setattr(self.flash_rect, "size", val),
            )
            self.add_widget(self.flash_widget)
        self.flash_widget.opacity = 1.0
        Animation(opacity=0.0, duration=0.3).start(self.flash_widget)
```

- [ ] **Step 3: Trigger the flash inside `capture_image`**

In `CameraScreen.capture_image`, after `cv2.imwrite(save_path, frame)` but before `Clock.unschedule(self.update_preview)`:

```python
        self._flash()
```

- [ ] **Step 4: Swap transition for the camera→insult hop**

Replace `CameraScreen.go_to_insult` with:

```python
    def go_to_insult(self, dt):
        prev = self.manager.transition
        self.manager.transition = SlideTransition(direction='left', duration=0.4)
        self.manager.current = 'insult'

        def _restore(_):
            self.manager.transition = prev
        Clock.schedule_once(_restore, 0.5)
```

- [ ] **Step 5: Verify**

`python -c "import ast; ast.parse(open('dissman.py').read())"` and `python -m pytest tests/ -q` (22 pass). Smoke test in Kivy: enter camera screen, observe white flash at capture and a slide animation to InsultScreen.

- [ ] **Step 6: Commit**

```bash
git add dissman.py
git commit -m "Add capture flash and slide transition from camera to insult"
```

---

## Task 7: Add `face.png` to TeachCategoryScreen

**Files:**
- Modify: `insultmaster3.kv` (TeachCategoryScreen block, lines 140-171)

- [ ] **Step 1: Edit the kv**

In `<TeachCategoryScreen>:`, insert an Image block between the "Teach Dissman an insult" Label and the "Which category?" Label:

```kv
        Image:
            source: 'face.png'
            size_hint_y: None
            height: 165
```

The block becomes:

```kv
        Label:
            text: 'Teach Dissman an insult'
            font_name: app.fonts['heading']
            font_size: '24sp'
            color: app.theme_colors['primary']
            size_hint_y: None
            height: 40
        Image:
            source: 'face.png'
            size_hint_y: None
            height: 165
        Label:
            text: 'Which category?'
            font_name: app.fonts['body']
            font_size: '18sp'
            color: app.theme_colors['secondary']
            size_hint_y: None
            height: 30
```

- [ ] **Step 2: Smoke-test**

Run the app, navigate to the teach flow, confirm `face.png` appears and the three category buttons still fit.

- [ ] **Step 3: Commit**

```bash
git add insultmaster3.kv
git commit -m "Add face.png to TeachCategoryScreen"
```

---

## Task 8: Update teach-keyboard prompts

**Files:**
- Modify: `dissman.py` (TeachAdjScreen and TeachNounScreen class attributes)

- [ ] **Step 1: Edit the strings**

In `dissman.py`, find:

```python
class TeachAdjScreen(TeachWordScreen):
    prompt = "Type the adjective"
    pos_key = "adj"
    next_screen = "teach_noun"


class TeachNounScreen(TeachWordScreen):
    prompt = "Type the noun"
    pos_key = "noun"
    next_screen = "teach_submit"
```

Change `prompt` strings to:

```python
class TeachAdjScreen(TeachWordScreen):
    prompt = "Type the insulting adjective"
    ...

class TeachNounScreen(TeachWordScreen):
    prompt = "Type the insulting noun"
    ...
```

- [ ] **Step 2: Commit**

```bash
git add dissman.py
git commit -m "Sharper wording on teach-keyboard prompts"
```

---

## Task 9: Extend TeachSubmitScreen delay

**Files:**
- Modify: `dissman.py` (`TeachSubmitScreen.on_enter`, line ~554 currently has `Clock.schedule_once(lambda dt: self._go_home(), 3)`).

The TTS line is `"Thanks. I will remember: {adj} {noun}."` — typically 3-4 seconds on `espeak-ng`. The current 3s schedule fires before speech finishes. Bump to 6s to leave a comfortable tail.

- [ ] **Step 1: Edit**

Change:

```python
        Clock.schedule_once(lambda dt: self._go_home(), 3)
```

to:

```python
        Clock.schedule_once(lambda dt: self._go_home(), 6)
```

(There are two `_go_home` schedules in `TeachSubmitScreen` — one in the success branch and one in the error branch. Only extend the success-branch one. Leave the error branch at 2s.)

- [ ] **Step 2: Commit**

```bash
git add dissman.py
git commit -m "Hold TeachSubmitScreen long enough to finish the 'I will remember' speech"
```

---

## Task 10: Create `requirements.txt` with pinned versions

**Files:**
- Create: `requirements.txt`
- Modify: `CLAUDE.md` — Dependencies section

We pin exact versions matching what the dev env (`C:\Users\Ryan\conda-envs\dissman`) successfully installed. The Pi may need slightly different versions (different python minor / different arch) — the Pi-specific deps (`gpiozero`, `python-escpos`) are listed but commented since they fail on Windows.

- [ ] **Step 1: Capture dev env versions**

Run:

```bash
"C:\Users\Ryan\conda-envs\dissman\python.exe" -m pip freeze > /tmp/freeze.txt
```

Inspect `/tmp/freeze.txt` and identify the *direct* dependencies (the ones we install explicitly). Transitive deps (httpcore, sniffio, jiter, etc.) are pulled in automatically — pin them too so a clean install is bit-identical.

The list of direct deps the app uses: `kivy`, `opencv-python`, `numpy`, `openai`, `python-dotenv`, `pillow`, `requests`, `pyttsx3`, `gpiozero` (Pi), `python-escpos` (Pi).

- [ ] **Step 2: Write `requirements.txt`**

Use the exact `==` versions seen in your `pip freeze` output. The list below is the expected shape based on the install log from earlier in the session — verify each version against your local freeze before committing:

```
# Dissman pinned dependencies. Verified working on Python 3.12 (Windows dev)
# and Python 3.11 (Raspberry Pi OS). Regenerate after upgrades with:
#   pip freeze > requirements.lock.txt
# then promote curated entries here.

# --- Core ---
kivy==2.3.1
opencv-python==4.13.0.92
numpy==2.4.6
pillow==12.2.0
requests==2.34.2
python-dotenv==1.2.2

# --- TTS ---
pyttsx3==2.99

# --- OpenAI SDK + transitive deps that affect API behavior ---
openai==2.37.0
httpx==0.28.1
pydantic==2.13.4

# --- Pi-only (uncomment on Raspberry Pi install) ---
# gpiozero==2.0.1
# python-escpos==3.1
```

(The Pi-only block stays commented because `gpiozero`/`python-escpos` have Linux-only runtime expectations and may break Windows installs. Pi install instructions live in CLAUDE.md.)

- [ ] **Step 3: Update `CLAUDE.md` Dependencies section**

Replace the current bare list:

```
**Dependencies (no requirements.txt exists):**
```
kivy, opencv-python, numpy, gpiozero, python-escpos, pyttsx3, openai, python-dotenv, pillow, requests
```
System packages on Pi: `espeak-ng`
```

with:

```
**Dependencies:** All versions pinned in `requirements.txt`.

Install (development, Windows/macOS):
```
python -m pip install -r requirements.txt
```

Install (Raspberry Pi):
```
python3 -m pip install -r requirements.txt
python3 -m pip install gpiozero==2.0.1 python-escpos==3.1
sudo apt install -y espeak-ng
```

The Pi-only deps live commented at the bottom of `requirements.txt` so they don't break Windows installs. Pin updates: install the new version locally, confirm Dissman still runs, then bump the `==` line in `requirements.txt`.
```

(Use the actual versions you pinned.)

- [ ] **Step 4: Verify install from scratch (optional but recommended)**

```bash
"C:\ProgramData\anaconda3\Scripts\conda.exe" create -n dissman-verify python=3.12 -y
"C:\Users\Ryan\conda-envs\dissman-verify\python.exe" -m pip install -r requirements.txt
"C:\Users\Ryan\conda-envs\dissman-verify\python.exe" -m pytest tests/ -q
```

Expected: tests pass. If any version is wrong (resolver complains), update the offending line.

Then optionally delete the verify env:

```bash
"C:\ProgramData\anaconda3\Scripts\conda.exe" env remove -n dissman-verify -y
```

- [ ] **Step 5: Commit**

```bash
git add requirements.txt CLAUDE.md
git commit -m "Pin all dependencies in requirements.txt; document install procedure"
```

---

## Task 11: Push to main and deploy to the Pi

- [ ] **Step 1: Push**

```bash
git push origin main
```

- [ ] **Step 2: Update Pi deps (one-time, after pulling)**

SSH to the Pi (the `start.sh` auto-pull will fetch the new code at next restart, but it won't reinstall pip packages — that's a manual step):

```bash
sudo systemctl stop dissman.service
cd /home/dissman/Documents/app
git pull
python3 -m pip install -r requirements.txt
python3 -m pip install gpiozero==2.0.1 python-escpos==3.1
sudo systemctl start dissman.service
sudo journalctl -u dissman.service -b --no-pager | tail -40
```

- [ ] **Step 3: Live test**

Insert a coin, walk through the full flow once. Verify:
- Category screen face is bigger.
- Camera flash visible at capture.
- Slide animation into insult screen.
- Teach flow: "Pick a category" shows face; "Type the insulting adjective" prompt; full "I will remember" speech finishes before the splash returns.
- Drive 10+ runs and confirm at least one Goya-variant doodle (or check `journalctl` for the `[image-gen] using prompt 'goya_saturn'` log line).

---

## Self-review checklist

- **Spec coverage:** UI items (image size, two animations, wording fixes, face on teach screen, keyboard prompts, longer delay) → Tasks 5, 6, 4, 7, 8, 9. Prompt CSV (externalize, weighted, base flag, extensible) → Tasks 1, 2, 3. Dependency pinning → Task 10. Deploy → Task 11.
- **No placeholders:** every code change has a literal target and replacement.
- **Open clarification:** Task 4 has a confirmation step rather than a guess.
- **Reversibility:** every change is a single commit; revert is `git revert <sha>`.

## Execution

Plan saved to `docs/superpowers/plans/2026-05-20-ui-polish-prompt-store-deps.md`. Two options:

1. **Subagent-Driven (recommended)** — dispatch one subagent per task with two-stage review.
2. **Inline Execution** — execute tasks in this session with checkpoints.

Which approach?
