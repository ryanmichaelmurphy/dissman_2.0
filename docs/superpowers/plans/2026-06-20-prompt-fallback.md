# Prompt Fallback Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When OpenAI content-rejects the chosen drawing prompt, automatically retry through a tiered chain (primary → is_base → last-successful → hardcoded happy prompt) so a coin insert still yields a printed doodle.

**Architecture:** A pure, unit-tested `prompt_fallback.py` (candidate assembly + last-successful persistence) plus a thin retry loop in `dissman.py`'s `_work()`. Everything operates on the existing `PromptChoice(name, prompt, settings)` so the winning tier's `PrintSettings` reach the printer via `job.settings`. Only `openai.BadRequestError` (400/moderation) advances the chain; any other error stops.

**Tech Stack:** Python 3.12, openai, Pillow, python-dotenv, pytest. All deps already pinned. Reuses `prompt_store.PromptChoice` and `print_pipeline.PrintSettings`.

## Global Constraints

- Source of truth for this work: `docs/superpowers/specs/2026-06-20-prompt-fallback-design.md` (commit `bade720`).
- Only `openai.BadRequestError` (HTTP 400, e.g. `moderation_blocked`) advances the chain. Network/5xx/403/timeout → log, `job.error = True`, stop. Verbatim final fallback text: `Make this person look really really happy and enthusiastic` (name `fallback_happy`).
- The unit of work is `prompt_store.PromptChoice(name, prompt, settings)`. Settings travel end-to-end; the winning tier sets `job.settings`.
- Last-successful state file: `prompts/.last_successful.json` — **git-ignored** (the Pi's `git reset --hard origin/main` would otherwise wipe it). The hardcoded final fallback is **never** recorded as last-successful.
- Run tests with the conda env python (system python lacks PIL): `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -m pytest`.
- Work on branch `feature/prompt-fallback` (already created off `origin/fallback`). Never commit half-done work to `main` — the Pi pulls `main` on reboot.

---

### Task 1: `PromptStore.base()` — expose the is_base row

**Files:**
- Modify: `prompt_store.py` (add `base` method to `PromptStore`)
- Modify: `tests/test_prompt_store.py`

**Interfaces:**
- Consumes: existing `PromptChoice`, `PromptStore._to_choice`, `_is_truthy`, `_load`.
- Produces: `PromptStore.base() -> PromptChoice | None` (the `is_base == true` row with parsed settings, or `None`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompt_store.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -m pytest tests/test_prompt_store.py -k base -v`
Expected: FAIL with `AttributeError: 'PromptStore' object has no attribute 'base'`

- [ ] **Step 3: Add the `base` method**

In `prompt_store.py`, add this method to `PromptStore` immediately after `choose` (before `_to_choice`):

```python
    def base(self):
        """Return the is_base==true row as a PromptChoice, or None if absent."""
        for r in self._load():
            if _is_truthy(r.get("is_base")):
                return self._to_choice(r)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -m pytest tests/test_prompt_store.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add prompt_store.py tests/test_prompt_store.py
git commit -m "Add PromptStore.base() to expose the is_base prompt with settings"
```

---

### Task 2: `prompt_fallback.py` — candidate chain + persistence

**Files:**
- Create: `prompt_fallback.py`
- Test: `tests/test_prompt_fallback.py`

**Interfaces:**
- Consumes: `prompt_store.PromptChoice`, `print_pipeline.PrintSettings`.
- Produces:
  - `FINAL_FALLBACK: str`, `FINAL_FALLBACK_NAME: str`
  - `Candidate` (NamedTuple `tier: str, choice: PromptChoice`)
  - `build_candidates(primary, is_base, last_successful, final_fallback) -> list[Candidate]` — inputs are `PromptChoice | None` (final always a `PromptChoice`); ordered, drops None/empty, de-dups by prompt text (earlier wins).
  - `read_last_successful(path) -> PromptChoice | None` (never raises; settings → defaults if absent/garbage)
  - `write_last_successful(path, choice: PromptChoice) -> None` (atomic, best-effort)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompt_fallback.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -m pytest tests/test_prompt_fallback.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prompt_fallback'`

- [ ] **Step 3: Write `prompt_fallback.py`**

```python
"""Prompt fallback chain for production image generation.

Pure helpers used by dissman.py's _work(): assemble an ordered, de-duplicated
list of prompt candidates, and persist/restore the last prompt that succeeded.
All functions operate on prompt_store.PromptChoice so each prompt's PrintSettings
travel with it. Persistence never raises — recording state must not break image
generation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple, Optional

from prompt_store import PromptChoice
from print_pipeline import PrintSettings

FINAL_FALLBACK = "Make this person look really really happy and enthusiastic"
FINAL_FALLBACK_NAME = "fallback_happy"


class Candidate(NamedTuple):
    tier: str
    choice: PromptChoice


def build_candidates(primary, is_base, last_successful, final_fallback):
    """Ordered, de-duped candidates. Inputs are PromptChoice | None (final not None)."""
    tiers = [
        ("primary", primary),
        ("is_base", is_base),
        ("last_successful", last_successful),
        ("final_fallback", final_fallback),
    ]
    out = []
    seen = set()
    for tier, choice in tiers:
        if choice is None:
            continue
        text = (choice.prompt or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(Candidate(tier, choice))
    return out


def read_last_successful(path) -> Optional[PromptChoice]:
    """Return the persisted PromptChoice, or None on missing/corrupt file."""
    try:
        data = json.loads(Path(path).read_text())
        name = data["name"]
        prompt = data["prompt"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    raw = data.get("settings")
    try:
        settings = PrintSettings(**raw) if isinstance(raw, dict) else PrintSettings.defaults()
    except TypeError:
        settings = PrintSettings.defaults()
    return PromptChoice(name, prompt, settings)


def write_last_successful(path, choice: PromptChoice) -> None:
    """Atomically persist the choice; best-effort (failures logged, swallowed)."""
    try:
        path = Path(path)
        payload = {
            "name": choice.name,
            "prompt": choice.prompt,
            "settings": asdict(choice.settings),
            "ts": int(time.time()),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, path)
    except OSError as e:
        print(f"[prompt_fallback] could not record last-successful: {e}", flush=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -m pytest tests/test_prompt_fallback.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Commit**

```bash
git add prompt_fallback.py tests/test_prompt_fallback.py
git commit -m "Add prompt_fallback: candidate chain + last-successful persistence"
```

---

### Task 3: Wire the retry loop into `dissman.py` + gitignore the state file

**Files:**
- Modify: `dissman.py` (imports near line 33 & line 175-179; `_work` ~134-159)
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `build_candidates`, `read_last_successful`, `write_last_successful`, `FINAL_FALLBACK`, `FINAL_FALLBACK_NAME` (Task 2); `PromptStore.base` (Task 1); `PromptChoice`, `PrintSettings`.
- No unit tests (Kivy module / real API). Verified by syntax check + manual run.

- [ ] **Step 1: Add `import openai`**

In `dissman.py`, after the line `from openai import OpenAI` (line 33), add:

```python
import openai
```

- [ ] **Step 2: Add fallback imports + state path**

Replace the existing block (lines 175-179):

```python
from insult_store import InsultStore, CATEGORY_LABELS

INSULT_STORE = InsultStore(BASE_DIR / "insults")
from prompt_store import PromptStore
PROMPT_STORE = PromptStore(BASE_DIR / "prompts" / "drawing_prompts.csv")
```

with:

```python
from insult_store import InsultStore, CATEGORY_LABELS

INSULT_STORE = InsultStore(BASE_DIR / "insults")
from prompt_store import PromptStore, PromptChoice
from print_pipeline import PrintSettings
from prompt_fallback import (
    build_candidates, read_last_successful, write_last_successful,
    FINAL_FALLBACK, FINAL_FALLBACK_NAME,
)
PROMPT_STORE = PromptStore(BASE_DIR / "prompts" / "drawing_prompts.csv")
LAST_SUCCESS_PATH = BASE_DIR / "prompts" / ".last_successful.json"
```

- [ ] **Step 3: Replace the `_work` function**

Replace the entire current `_work` (from `def _work():` through `job.error = True` at the end of its `except`) with:

```python
    def _work():
        final_choice = PromptChoice(
            FINAL_FALLBACK_NAME, FINAL_FALLBACK, PrintSettings.defaults()
        )
        candidates = build_candidates(
            primary=PROMPT_STORE.choose(),
            is_base=PROMPT_STORE.base(),
            last_successful=read_last_successful(LAST_SUCCESS_PATH),
            final_fallback=final_choice,
        )

        for tier, choice in candidates:
            print(f"[image-gen] attempt tier={tier} name='{choice.name}'", flush=True)
            try:
                with open(source_image_path, "rb") as f:
                    response = client.images.edit(
                        model="gpt-image-1", image=f, prompt=choice.prompt,
                        n=1, size="1024x1024",
                    )
                data = base64.b64decode(response.data[0].b64_json)
                with open(out_path, "wb") as f:
                    f.write(data)
                job.image_path = out_path
                job.settings = choice.settings
                job.ready = True
                if tier != "final_fallback":
                    write_last_successful(LAST_SUCCESS_PATH, choice)
                print(f"[image-gen] success tier={tier} name='{choice.name}'", flush=True)
                return
            except openai.BadRequestError as e:
                # content/moderation rejection — a different prompt may pass
                detail = getattr(e, "body", None) or str(e)
                print(f"[image-gen] rejected tier={tier} name='{choice.name}' "
                      f"(status={getattr(e, 'status_code', None)}): {detail}", flush=True)
                continue
            except Exception as e:
                # network/5xx/403/timeout — a different prompt cannot help; stop
                detail = getattr(e, "body", None) or str(e)
                print(f"[image-gen] failed (non-content) tier={tier} "
                      f"name='{choice.name}': {type(e).__name__}: {detail}", flush=True)
                job.error = True
                return

        print("[image-gen] all candidates rejected; giving up", flush=True)
        job.error = True
```

- [ ] **Step 4: Add the state file to `.gitignore`**

Append to `.gitignore`:

```gitignore

# Runtime: last prompt that passed moderation (survives the boot reset --hard)
prompts/.last_successful.json
```

- [ ] **Step 5: Syntax check + import-resolution smoke test**

Run: `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -c "import ast; ast.parse(open('dissman.py').read()); print('dissman.py parses OK')"`
Expected: `dissman.py parses OK`

Run (verifies the new modules import together without Kivy):
```bash
& "C:\Users\Ryan\conda-envs\dissman\python.exe" -c "import prompt_fallback, prompt_store, print_pipeline; import openai; print('openai has BadRequestError:', hasattr(openai, 'BadRequestError'))"
```
Expected: `openai has BadRequestError: True`

- [ ] **Step 6: Commit**

```bash
git add dissman.py .gitignore
git commit -m "Wire prompt fallback chain into image generation _work()"
```

**Manual verification (real hardware / live run on the Pi):** with the current abuse-coded primary prompt, a coin insert should now log `attempt tier=primary … rejected … → attempt tier=is_base …` etc. in the journal and ultimately print *something* (the happy fallback if all CSV prompts are blocked), instead of bouncing to Splash. Confirm `prompts/.last_successful.json` appears after a non-fallback success and is not tracked by git.

---

### Task 4: Documentation

**Files:**
- Modify: `CLAUDE.md`

**Interfaces:** none.

- [ ] **Step 1: Document the fallback chain**

In CLAUDE.md, under the drawing-prompts / integration section, add a bullet group: image generation retries through `primary → is_base → last-successful → hardcoded "happy" fallback` on OpenAI **content** rejections only (other errors stop); the chain lives in `prompt_fallback.py` (pure, unit-tested) + the `_work()` loop; the last prompt that passed moderation is persisted to the **git-ignored** `prompts/.last_successful.json` (atomic write, survives reboots, like `boot-update.log`); the hardcoded fallback is never recorded. Add `prompt_fallback.py` to the pytest-covered pure-module list.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Document the prompt fallback chain"
```

---

## Self-Review

**Spec coverage (`bade720`):**
- Tiered chain primary→is_base→last_successful→final, content-rejection-only advance → Task 3 loop. ✓
- `prompt_fallback.py` (`build_candidates`, `read/write_last_successful`, `FINAL_FALLBACK*`, `Candidate`) → Task 2. ✓
- Settings travel end-to-end; winning tier sets `job.settings`; final fallback uses defaults → Task 2 (`PromptChoice` candidates) + Task 3 (`job.settings = choice.settings`). ✓
- `PromptStore.base() -> PromptChoice | None` → Task 1. ✓
- Persisted JSON with `settings` block, degrades to defaults if absent/garbage; atomic write → Task 2 (`read/write_last_successful`), tested. ✓
- `.gitignore` gains one line (not a new file) → Task 3 Step 4 (appends to existing). ✓
- Dedup by prompt text, earlier tier (and settings) wins; final dropped if duplicate → Task 2, tested. ✓
- Only `BadRequestError` advances; other errors stop; all-rejected → `job.error` → Task 3 loop. ✓
- `USE_REAL_API` bypass unchanged → Task 3 leaves the bypass branch untouched. ✓

**Placeholder scan:** none — every code/test step is complete.

**Type consistency:** `PromptChoice(name, prompt, settings)` and `PrintSettings` used identically across Tasks 1–3. `Candidate(tier, choice)` defined in Task 2, unpacked as `for tier, choice in candidates` in Task 3. `build_candidates`/`read_last_successful`/`write_last_successful` signatures match between `prompt_fallback.py` and `tests/test_prompt_fallback.py` and the Task 3 call sites. `PrintSettings(**raw)` guarded by `try/except TypeError` (extra keys) — note partial dicts are fine because every `PrintSettings` field has a default.

**Note on `read_last_successful` robustness:** `PrintSettings(**raw)` raises `TypeError` on unexpected keys (caught → defaults). Missing keys do NOT raise (dataclass defaults fill them), which is the intended "degrade gracefully" behavior.
