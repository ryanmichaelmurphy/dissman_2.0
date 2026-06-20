# Print-Tuning & Prompt-Probing Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone suite to iterate prompts past OpenAI moderation and tune per-prompt thermal print variables, with a shared rendering pipeline so tuned settings == production output.

**Architecture:** A pure `print_pipeline.py` (PrintSettings + render_for_thermal) is imported by both a new `prompt_testing/lab.py` CLI and the existing `dissman.py`. The prompt CSV gains print-variable columns; `PromptStore.choose()` carries them through to print time. A gitignored cache decouples the paid API call from free print-tuning reprints.

**Tech Stack:** Python 3.12, Pillow (PIL), python-escpos, openai, python-dotenv, pytest. All deps already pinned in `requirements.txt`.

## Global Constraints

- Pure-stdlib or already-pinned deps only — no new pip installs (Pi auto-deploys by git-pull, no SSH-only install). Verbatim: PIL, openai, python-escpos, python-dotenv are already in `requirements.txt`.
- `print_pipeline.py` top level must import without PIL (PIL imported lazily inside `render_for_thermal`) so `prompt_store` stays PIL-free at import.
- CSV column order (verbatim): `name,weight,is_base,binarization,threshold,contrast,brightness,resize_width,sharpness,gamma,prompt`
- Defaults that preserve today's behavior (verbatim): `binarization=dither, threshold=128, contrast=0.3, brightness=1.0, resize_width=380, sharpness=1.0, gamma=1.0`.
- Run tests with: `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -m pytest` on Windows (Kivy env). Plain `python -m pytest` also works for the pure modules.
- Work happens on branch `feature/print-tuning-suite` (already created). Never commit half-done work to `main` — the Pi pulls `main` on reboot.

---

### Task 1: `print_pipeline.py` — PrintSettings + render_for_thermal

**Files:**
- Create: `print_pipeline.py`
- Test: `tests/test_print_pipeline.py`

**Interfaces:**
- Produces: `PrintSettings` (frozen dataclass; fields `binarization:str, threshold:int, contrast:float, brightness:float, resize_width:int, sharpness:float, gamma:float`); `PrintSettings.defaults() -> PrintSettings`; `PrintSettings.from_row(row: dict) -> PrintSettings`; `render_for_thermal(image: PIL.Image.Image, settings: PrintSettings) -> PIL.Image.Image` (returns mode `"1"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_print_pipeline.py
from PIL import Image

from print_pipeline import PrintSettings, render_for_thermal


def _ramp(width=256, height=10):
    """L-mode image where column x has gray value x."""
    img = Image.new("L", (width, height))
    img.putdata([x for _ in range(height) for x in range(width)])
    return img


def test_defaults_match_todays_behavior():
    s = PrintSettings.defaults()
    assert s.binarization == "dither"
    assert s.threshold == 128
    assert s.contrast == 0.3
    assert s.brightness == 1.0
    assert s.resize_width == 380
    assert s.sharpness == 1.0
    assert s.gamma == 1.0


def test_from_row_parses_and_clamps():
    s = PrintSettings.from_row({
        "binarization": "THRESHOLD", "threshold": "300", "contrast": "1.5",
        "brightness": "0.9", "resize_width": "384", "sharpness": "2.0",
        "gamma": "1.2",
    })
    assert s.binarization == "threshold"
    assert s.threshold == 255          # clamped 0-255
    assert s.contrast == 1.5
    assert s.resize_width == 384


def test_from_row_uses_defaults_for_missing_or_bad():
    s = PrintSettings.from_row({"binarization": "nonsense", "contrast": "oops"})
    assert s.binarization == "dither"  # unknown mode -> default
    assert s.contrast == 0.3           # unparseable -> default
    assert s.threshold == 128


def test_render_returns_one_bit_image():
    out = render_for_thermal(_ramp(), PrintSettings.defaults())
    assert out.mode == "1"


def test_threshold_binarization_splits_at_cutoff():
    s = PrintSettings(binarization="threshold", threshold=128, contrast=1.0,
                      brightness=1.0, resize_width=256, sharpness=1.0, gamma=1.0)
    out = render_for_thermal(_ramp(256, 10), s)
    assert out.getpixel((50, 5)) == 0      # below cutoff -> black
    assert out.getpixel((200, 5)) == 255   # above cutoff -> white


def test_dither_produces_only_black_and_white():
    gray = Image.new("L", (64, 64), 128)
    s = PrintSettings(binarization="dither", threshold=128, contrast=1.0,
                      brightness=1.0, resize_width=64, sharpness=1.0, gamma=1.0)
    out = render_for_thermal(gray, s)
    values = set(out.getdata())
    assert values <= {0, 255}
    assert 0 in values and 255 in values   # mid-gray actually dithers


def test_resize_width_is_honored():
    s = PrintSettings.from_row({"resize_width": "200"})
    out = render_for_thermal(_ramp(512, 512), s)
    assert out.width == 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_print_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'print_pipeline'`

- [ ] **Step 3: Write the implementation**

```python
# print_pipeline.py
"""Thermal print rendering, shared by the lab CLI and dissman.py.

PrintSettings mirrors the per-prompt columns in prompts/drawing_prompts.csv.
render_for_thermal turns an RGB/L image into the 1-bit (mode "1") image the
POS-5890 prints, so what the lab tunes is exactly what production prints.

PIL is imported lazily inside render_for_thermal so importing PrintSettings
(e.g. from prompt_store) does not require Pillow.
"""

from __future__ import annotations

from dataclasses import dataclass

_BINARIZATION_MODES = ("dither", "threshold")


@dataclass(frozen=True)
class PrintSettings:
    binarization: str = "dither"
    threshold: int = 128
    contrast: float = 0.3
    brightness: float = 1.0
    resize_width: int = 380
    sharpness: float = 1.0
    gamma: float = 1.0

    @classmethod
    def defaults(cls) -> "PrintSettings":
        return cls()

    @classmethod
    def from_row(cls, row: dict) -> "PrintSettings":
        def _f(key, default, lo, hi):
            try:
                v = float(row.get(key, ""))
            except (TypeError, ValueError):
                return default
            return max(lo, min(hi, v))

        def _i(key, default, lo, hi):
            return int(round(_f(key, default, lo, hi)))

        mode = str(row.get("binarization", "")).strip().lower()
        if mode not in _BINARIZATION_MODES:
            mode = "dither"

        return cls(
            binarization=mode,
            threshold=_i("threshold", 128, 0, 255),
            contrast=_f("contrast", 0.3, 0.0, 3.0),
            brightness=_f("brightness", 1.0, 0.0, 2.0),
            resize_width=_i("resize_width", 380, 1, 384),
            sharpness=_f("sharpness", 1.0, 0.0, 3.0),
            gamma=_f("gamma", 1.0, 0.1, 3.0),
        )


def render_for_thermal(image, settings: PrintSettings):
    """Return a mode "1" image ready for escpos p.image()."""
    from PIL import Image, ImageEnhance

    img = image.convert("L")

    w = max(1, int(settings.resize_width))
    h = max(1, round(img.height * w / img.width))
    img = img.resize((w, h), Image.Resampling.LANCZOS)

    if settings.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(settings.contrast)
    if settings.brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(settings.brightness)
    if settings.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(settings.sharpness)
    if settings.gamma != 1.0:
        g = settings.gamma
        lut = [int(round(255 * (i / 255) ** g)) for i in range(256)]
        img = img.point(lut)

    if settings.binarization == "threshold":
        t = settings.threshold
        img = img.point(lambda p, t=t: 255 if p >= t else 0)
        img = img.convert("1", dither=Image.Dither.NONE)
    else:
        img = img.convert("1")  # Floyd–Steinberg (PIL default)

    return img
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_print_pipeline.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add print_pipeline.py tests/test_print_pipeline.py
git commit -m "Add print_pipeline: PrintSettings + render_for_thermal"
```

---

### Task 2: Expand `drawing_prompts.csv` with print-variable columns

**Files:**
- Modify: `prompts/drawing_prompts.csv`

**Interfaces:**
- Produces: CSV rows with columns `name,weight,is_base,binarization,threshold,contrast,brightness,resize_width,sharpness,gamma,prompt`. Consumed by Task 3 (`PromptStore`) and the Task 5 CLI.

- [ ] **Step 1: Rewrite the CSV with new columns, back-filling defaults**

Replace the entire file with (prompt text preserved verbatim from the current rows):

```csv
name,weight,is_base,binarization,threshold,contrast,brightness,resize_width,sharpness,gamma,prompt
middle_school_bully,9,true,dither,128,0.3,1.0,380,1.0,1.0,"You are a middle school bully. Draw this person as a crude middle school notebook doodle. Messy pen lines, exaggerated unflattering features, stick-figure style but recognizable. Make them uglier than they actually are with a stupid facial expression."
goya_saturn,1,false,dither,128,0.3,1.0,380,1.0,1.0,"Draw the person in the image as a hideous old person in the style of Francisco de Goya's Black Paintings, in particular the painting Saturn Devouring His Son."
```

- [ ] **Step 2: Verify it parses and round-trips**

Run:
```bash
python -c "import csv; rows=list(csv.DictReader(open('prompts/drawing_prompts.csv',newline=''))); print(len(rows), rows[0]['binarization'], rows[0]['contrast']); assert rows[0]['prompt'].startswith('You are a middle school bully')"
```
Expected: `2 dither 0.3`

- [ ] **Step 3: Commit**

```bash
git add prompts/drawing_prompts.csv
git commit -m "Add per-prompt print-variable columns to drawing_prompts.csv"
```

---

### Task 3: `PromptStore.choose()` carries print settings

**Files:**
- Modify: `prompt_store.py`
- Modify: `tests/test_prompt_store.py`

**Interfaces:**
- Consumes: `PrintSettings`, `PrintSettings.from_row`, `PrintSettings.defaults` from Task 1.
- Produces: `PromptChoice` (frozen dataclass; `name:str, prompt:str, settings:PrintSettings`). `PromptStore.choose() -> PromptChoice`.

- [ ] **Step 1: Update the existing tests for the new return type**

In `tests/test_prompt_store.py`, change the import line and every `choose()` call site. Replace tuple unpacking with attribute access:

- Line 7 import: `from prompt_store import PromptStore, PromptChoice, FALLBACK_PROMPT`
- `test_choose_returns_prompt_text`: replace body assertions with
```python
    c = s.choose()
    assert c.name == "a"
    assert c.prompt == "the only prompt"
```
- `test_choose_uses_weights`: `picks = [s.choose().name for _ in range(1000)]`
- `test_choose_falls_back_to_base_when_all_weights_zero`:
```python
    c = s.choose()
    assert c.name == "base"
    assert c.prompt == "base prompt"
```
- `test_choose_falls_back_to_hardcoded_when_csv_missing`:
```python
    c = s.choose()
    assert c.name == "fallback"
    assert c.prompt == FALLBACK_PROMPT
```
- `test_choose_falls_back_to_hardcoded_when_csv_empty`: `assert s.choose().name == "fallback"`
- `test_negative_weights_treated_as_zero`: `assert s.choose().name == "base"`
- `test_non_numeric_weight_treated_as_zero`: `assert s.choose().name == "base"`

- [ ] **Step 2: Add a test asserting settings come through**

Append to `tests/test_prompt_store.py`:

```python
from print_pipeline import PrintSettings


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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_prompt_store.py -v`
Expected: FAIL with `ImportError: cannot import name 'PromptChoice'`

- [ ] **Step 4: Update `prompt_store.py`**

Update the module docstring's column line to the new schema. Add imports and the dataclass below the existing imports (after line 17):

```python
from dataclasses import dataclass

from print_pipeline import PrintSettings


@dataclass(frozen=True)
class PromptChoice:
    name: str
    prompt: str
    settings: PrintSettings
```

Replace the entire `choose` method body with:

```python
    def choose(self):
        rows = self._load()
        if not rows:
            return PromptChoice("fallback", FALLBACK_PROMPT, PrintSettings.defaults())

        weighted = [(r, _parse_weight(r.get("weight"))) for r in rows]
        positive = [(r, w) for r, w in weighted if w > 0]

        if positive:
            choice = random.choices(
                [r for r, _ in positive],
                weights=[w for _, w in positive],
                k=1,
            )[0]
            return self._to_choice(choice)

        for r in rows:
            if _is_truthy(r.get("is_base")):
                return self._to_choice(r)

        return PromptChoice("fallback", FALLBACK_PROMPT, PrintSettings.defaults())

    @staticmethod
    def _to_choice(row):
        return PromptChoice(
            row.get("name", ""),
            row.get("prompt", ""),
            PrintSettings.from_row(row),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_prompt_store.py tests/test_print_pipeline.py -v`
Expected: PASS (all green)

- [ ] **Step 6: Commit**

```bash
git add prompt_store.py tests/test_prompt_store.py
git commit -m "PromptStore.choose returns PromptChoice with print settings"
```

---

### Task 4: Wire settings through `dissman.py` to the printer

**Files:**
- Modify: `dissman.py` (ImageJob ~99-108; `_work` ~133-135; `print_image_and_text` ~441-457)

**Interfaces:**
- Consumes: `PromptStore.choose() -> PromptChoice` (Task 3); `render_for_thermal`, `PrintSettings` (Task 1).
- No unit tests (Kivy UI / hardware). Verified by syntax check + manual run.

- [ ] **Step 1: Add `settings` to ImageJob**

In `class ImageJob`, add `self.settings = None` to BOTH `__init__` and `reset` (alongside `self.image_path = None`):

```python
    def __init__(self):
        self.image_path = None
        self.ready = False
        self.error = False
        self.settings = None

    def reset(self):
        self.image_path = None
        self.ready = False
        self.error = False
        self.settings = None
```

- [ ] **Step 2: Stash settings in `_work`**

Replace lines 133-135 (the `_work` head) with:

```python
    def _work():
        choice = PROMPT_STORE.choose()
        prompt_name, prompt_text = choice.name, choice.prompt
        job.settings = choice.settings
        print(f"[image-gen] using prompt '{prompt_name}'")
```

- [ ] **Step 3: Use render_for_thermal in the print path**

In `print_image_and_text`, replace the import line and the load/resize/contrast/save block (currently lines 442-457) with:

```python
        from PIL import Image
        from kivy.app import App
        from print_pipeline import render_for_thermal, PrintSettings

        # Load and process the image through the shared thermal pipeline so the
        # output matches what the lab tuned. Settings ride on the image job;
        # fall back to defaults (bypass mode / missing job).
        app = App.get_running_app()
        settings = getattr(app.image_job, "settings", None) or PrintSettings.defaults()
        image = Image.open(image_path)
        processed = render_for_thermal(image, settings)

        # Save the processed image to a temporary path
        timestamp = str(int(time.time()))
        temp_image_path = path + "downloaded_image_" + timestamp + ".png"
        processed.save(temp_image_path)
```

(Leave the `try/except/finally` printing block below unchanged — it still does `p.image(temp_image_path)` and the cleanup glob still matches `downloaded_image_`.)

- [ ] **Step 4: Syntax-check**

Run: `python -c "import ast; ast.parse(open('dissman.py').read()); print('dissman.py parses OK')"`
Expected: `dissman.py parses OK`

- [ ] **Step 5: Smoke-test the pipeline import wiring (no Kivy)**

Run:
```bash
python -c "from print_pipeline import render_for_thermal, PrintSettings; from PIL import Image; out=render_for_thermal(Image.new('RGB',(1024,1024),'gray'), PrintSettings.defaults()); print('render OK', out.mode, out.size)"
```
Expected: `render OK 1 (380, 380)`

- [ ] **Step 6: Commit**

```bash
git add dissman.py
git commit -m "Drive thermal printing from per-prompt PrintSettings"
```

**Manual verification (note in commit / PR, run later on real hardware):** with `DISSMAN_USE_API=1` or on the Pi, run the app end-to-end and confirm an insult still prints; the printed image should look identical to before (defaults preserve old behavior) until a row is tuned.

---

### Task 5: `prompt_testing/lab.py` CLI + `.gitignore`

**Files:**
- Create: `prompt_testing/lab.py`
- Create: `.gitignore`
- Test: `tests/test_lab.py`

**Interfaces:**
- Consumes: `PrintSettings`, `render_for_thermal` (Task 1); the CSV schema (Task 2).
- Produces: `cache_is_fresh(sidecar: dict | None, prompt_text: str) -> bool`; `load_row(rows: list[dict], name: str) -> dict | None` (testable helpers). CLI `main()` is manual-verify.

- [ ] **Step 1: Write failing tests for the pure helpers**

```python
# tests/test_lab.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "prompt_testing"))

from lab import cache_is_fresh, load_row


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lab.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lab'`

- [ ] **Step 3: Write `prompt_testing/lab.py`**

```python
#!/usr/bin/env python3
"""Prompt-probing and print-tuning lab for Dissman.

Two modes against ONE prompt row of prompts/drawing_prompts.csv:

  --test prompt : call the OpenAI image edit on test_photo.png, cache the raw
                  PNG + a JSON sidecar, and report accepted / rejected (+ the
                  exact moderation .body). Auto-regenerates when the row's
                  prompt text changed since the cached run; --force always does.

  --test print  : render the cached image through that row's print variables
                  and print it on the POS-5890. --preview writes a PNG instead.

The realistic Pi workflow (printer is owned by dissman.service):
  sudo systemctl stop dissman
  export OPENAI_API_KEY=...        # or: set -a; . /home/dissman/Documents/app/.env; set +a
  python3 prompt_testing/lab.py --name middle_school_bully --test prompt
  python3 prompt_testing/lab.py --name middle_school_bully --test print
  # edit prompts/drawing_prompts.csv, reprint, repeat; then:
  git commit -am "tune middle_school_bully" && sudo systemctl start dissman
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from print_pipeline import PrintSettings, render_for_thermal  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
PREVIEW_DIR = HERE / "preview"
DEFAULT_INPUT = HERE / "test_photo.png"
DEFAULT_CSV = ROOT / "prompts" / "drawing_prompts.csv"


def load_row(rows, name):
    for r in rows:
        if r.get("name") == name:
            return r
    return None


def cache_is_fresh(sidecar, prompt_text):
    """A cache entry is fresh only if it was ACCEPTED and the prompt is unchanged."""
    if not sidecar:
        return False
    if sidecar.get("status") != "accepted":
        return False
    return sidecar.get("prompt") == prompt_text


def _read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _sidecar_path(name):
    return CACHE_DIR / f"{name}.json"


def _image_path(name):
    return CACHE_DIR / f"{name}.png"


def _load_sidecar(name):
    p = _sidecar_path(name)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _write_sidecar(name, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _sidecar_path(name).write_text(json.dumps(data, indent=2))


def do_prompt(row, input_path, force):
    name = row["name"]
    prompt_text = row["prompt"]
    sidecar = _load_sidecar(name)

    if not force and cache_is_fresh(sidecar, prompt_text) and _image_path(name).exists():
        print(f"[lab] '{name}': cached image is fresh; use --force to regenerate.")
        return 0

    from openai import OpenAI
    from dotenv import load_dotenv

    load_dotenv()
    client = OpenAI()

    print(f"[lab] '{name}': calling images.edit on {input_path.name} ...")
    try:
        with open(input_path, "rb") as f:
            resp = client.images.edit(
                model="gpt-image-1", image=f, prompt=prompt_text, n=1,
                size="1024x1024",
            )
        data = base64.b64decode(resp.data[0].b64_json)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _image_path(name).write_bytes(data)
        _write_sidecar(name, {
            "name": name, "prompt": prompt_text, "input": input_path.name,
            "model": "gpt-image-1", "status": "accepted", "ts": int(time.time()),
        })
        print(f"[lab] '{name}': ACCEPTED -> cached {_image_path(name)}")
        return 0
    except Exception as e:  # noqa: BLE001 — we want the raw reason
        detail = getattr(e, "body", None) or str(e)
        status = getattr(e, "status_code", None)
        _write_sidecar(name, {
            "name": name, "prompt": prompt_text, "input": input_path.name,
            "model": "gpt-image-1", "status": "rejected",
            "error_type": type(e).__name__, "http_status": status,
            "error_body": detail, "ts": int(time.time()),
        })
        print(f"[lab] '{name}': REJECTED ({type(e).__name__}, status={status})")
        print(f"      {detail}")
        return 1


def do_print(row, preview):
    name = row["name"]
    img_path = _image_path(name)
    if not img_path.exists():
        print(f"[lab] '{name}': no cached image. Run --test prompt first.")
        return 1

    from PIL import Image

    settings = PrintSettings.from_row(row)
    label = (f"mode={settings.binarization} t={settings.threshold} "
             f"c={settings.contrast} b={settings.brightness} "
             f"w={settings.resize_width} s={settings.sharpness} g={settings.gamma}")
    processed = render_for_thermal(Image.open(img_path), settings)

    if preview:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out = PREVIEW_DIR / f"{name}.png"
        processed.save(out)
        print(f"[lab] '{name}': preview written {out}  [{label}]")
        return 0

    from escpos.printer import Usb

    p = Usb(0x0416, 0x5011, in_ep=0x81, out_ep=0x01, profile="POS-5890")
    tmp = CACHE_DIR / f"_print_{name}.png"
    processed.save(tmp)
    try:
        p._raw(b"\x1b\x40")
        p.text(f"\n{name}\n{label}\n\n")
        p.image(str(tmp))
        p.text("\n\n\n")
        p._raw(b"\x1b\x40")
        p.cut()
        print(f"[lab] '{name}': printed  [{label}]")
    finally:
        if tmp.exists():
            tmp.unlink()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dissman prompt/print lab")
    ap.add_argument("--name", required=True, help="prompt row 'name' to target")
    ap.add_argument("--test", required=True, choices=["prompt", "print"])
    ap.add_argument("--preview", action="store_true",
                    help="print mode: write a PNG instead of printing")
    ap.add_argument("--force", action="store_true",
                    help="prompt mode: regenerate even if cache is fresh")
    ap.add_argument("--input", default=str(DEFAULT_INPUT),
                    help="source photo for prompt mode")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args(argv)

    rows = _read_rows(args.csv)
    row = load_row(rows, args.name)
    if row is None:
        names = ", ".join(r.get("name", "") for r in rows)
        print(f"[lab] no row named '{args.name}'. Available: {names}")
        return 2

    if args.test == "prompt":
        return do_prompt(row, Path(args.input), args.force)
    return do_print(row, args.preview)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lab.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Create `.gitignore`**

```gitignore
# Local image-lab artifacts — never committed (per-machine, binary churn)
prompt_testing/cache/
prompt_testing/preview/

# Python bytecode
__pycache__/
*.pyc
```

- [ ] **Step 6: Verify the CLI wires up (no API, no printer)**

Run: `python prompt_testing/lab.py --name middle_school_bully --test print`
Expected: `[lab] 'middle_school_bully': no cached image. Run --test prompt first.` (exit 1 — expected, proves arg parsing + row lookup + path logic work without hardware)

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add prompt_testing/lab.py tests/test_lab.py .gitignore
git commit -m "Add prompt_testing/lab.py CLI and .gitignore for cache"
```

---

### Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md` (Data storage / PromptStore section)
- Modify: `README.md` (data-curation section)

**Interfaces:** none (docs).

- [ ] **Step 1: Update CLAUDE.md**

In the `prompts/` description and `PromptStore` bullets, document: the new CSV columns and their meaning; that `choose()` returns a `PromptChoice(name, prompt, settings)`; that `print_pipeline.render_for_thermal` is the shared renderer used by both `dissman.py` and `prompt_testing/lab.py`; and the lab workflow (stop service → probe/print → tune CSV → commit → start service). Add `print_pipeline.py` to the list of pytest-covered pure modules.

- [ ] **Step 2: Update README.md**

In "Data: how to add or change insults" / drawing-prompts section, note the new per-prompt print columns and point operators at `prompt_testing/lab.py` for tuning, with the one-gotcha (stop `dissman.service` to free the printer).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "Document print-tuning suite and expanded prompt CSV"
```

---

## Self-Review

**Spec coverage:**
- Two-stage cache-bridged workflow → Tasks 5 (CLI) + cache helpers. ✓
- `print_pipeline.py` shared renderer → Task 1, consumed in Tasks 4 & 5. ✓
- Auto-invalidating sidecar → `cache_is_fresh` (Task 5), tested. ✓
- CSV schema + backfill → Task 2. ✓
- `PromptStore.choose()` carries settings → Task 3. ✓
- dissman.py wiring (job stash + render) → Task 4. ✓
- Pi execution / printer contention → documented in `lab.py` docstring + Task 6. ✓
- Testing (`test_print_pipeline`, settings parsing) → Tasks 1, 3, 5. ✓

**Placeholder scan:** none — every code/test step shows full content.

**Type consistency:** `PrintSettings` fields and `from_row`/`defaults` names match across Tasks 1/3/4/5. `PromptChoice(name, prompt, settings)` consistent in Tasks 3 & 4. `cache_is_fresh`/`load_row` signatures match between `lab.py` and `test_lab.py`.

**Note on `render_for_thermal` + escpos:** Task 5 prints by saving the 1-bit image to PNG and passing the path to `p.image()` (mirrors today's dissman flow). Confirm on real hardware that escpos does not re-dither the already-1-bit PNG; if it does, pass the PIL image directly or set `impl`/`high_density_*`. Flagged as manual hardware verification.
