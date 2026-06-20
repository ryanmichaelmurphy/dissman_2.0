# Print-Tuning & Prompt-Probing Suite — Design

Date: 2026-06-20
Status: Approved (brainstorming)

## Problem

Two production issues with image prompting:

1. **OpenAI moderation rejects the drawing prompt.** Production is failing 100%
   with `moderation_blocked (safety_violations=[abuse])` on the
   `middle_school_bully` prompt (see commit `bb52a76`). There is no offline way
   to iterate prompt wording against the moderation endpoint.
2. **Returned images don't print well.** The thermal pipeline is hardcoded
   (`resize(380×380)` + `ImageEnhance.Contrast(0.3)` + escpos internal dither),
   with no way to tune per-prompt for the 1-bit POS-5890 head.

We want a standalone suite (outside `dissman.py`) to (a) test whether a prompt
passes moderation and (b) tune print variables per prompt, with the winning
variables flowing back into production via the prompt spreadsheet.

## Core principle: one API call → many prints

Generation costs money and can be moderation-rejected; printing is free and
repeatable. The suite separates the two stages and bridges them with a local
image cache, so print-variable tuning never re-spends the API.

## Components

### 1. `print_pipeline.py` (repo root, unit-tested)

Single source of truth for thermal rendering, imported by **both** the lab CLI
and `dissman.py` — so what you tune is byte-for-byte what production prints.

```
render_for_thermal(image: PIL.Image, settings: PrintSettings) -> PIL.Image  # mode "1"
```

Pipeline order: `resize_width` → `contrast` → `brightness` → `sharpness` →
`gamma` → binarize (`threshold` or `dither`). Returns a 1-bit (`mode "1"`)
image so escpos prints it as-is without re-dithering.

`PrintSettings` is a small dataclass mirroring the CSV columns, with a
`from_row(dict)` constructor that applies defaults/validation.

### 2. `prompt_testing/lab.py` (CLI; lives with `test_photo.png`)

Targets one prompt row by name, with a mode flag:

- `--name <row> --test prompt`: call `images.edit()` on `test_photo.png`, cache
  the raw PNG + a JSON sidecar, print **accepted / rejected + exact `.body`**.
- `--name <row> --test print`: load the cached image, run `render_for_thermal`
  with that row's settings, print on the POS-5890.
  - `--preview`: write the 1-bit PNG to `prompt_testing/preview/` instead of
    printing (no paper, works off-Pi).
- `--force`: regenerate the cached image even if fresh.

Tuning loop: edit the row's variables in `drawing_prompts.csv` → re-run
`--test print` → eyeball → repeat. Winners recorded by **manual edit** of the
CSV (matches the repo's "curation is manual" philosophy).

### 3. The cache (`prompt_testing/cache/`, gitignored)

- `cache/<name>.png` — the **raw** 1024×1024 API output (before any print
  processing, so adjustments never compound).
- `cache/<name>.json` — sidecar: exact prompt text used, input filename, model,
  timestamp, status (`accepted`/`rejected`), and error `.body` on rejection.
- **Rejections cache the sidecar only** (no image), giving a durable record of
  which wordings failed and why.
- **Auto-invalidation:** `--test prompt` regenerates when the row's current
  prompt text differs from the sidecar's (you edit wording constantly while
  dodging moderation). Unchanged text + existing image → reuse. `--force`
  overrides.

## CSV schema (`prompts/drawing_prompts.csv`)

Columns, print-vars ordered most → least impactful, long `prompt` kept last:

```
name,weight,is_base,binarization,threshold,contrast,brightness,resize_width,sharpness,gamma,prompt
```

| # | Column | Range | Default | Role |
|---|---|---|---|---|
| 1 | `binarization` | `threshold` \| `dither` | `dither` | gray→B/W; biggest lever |
| 2 | `threshold` | 0–255 (useful 90–180) | `128` | cutoff (threshold mode only) |
| 3 | `contrast` | 0.0–3.0 (useful 0.5–2.0) | `0.3` | PIL contrast (today's value) |
| 4 | `brightness` | 0.0–2.0 (useful 0.7–1.4) | `1.0` | PIL brightness |
| 5 | `resize_width` | 128–384 px | `380` | line weight; POS-5890 ≈ 384 dots |
| 6 | `sharpness` | 0.0–3.0 (useful 1.0–2.5) | `1.0` | edge crispness |
| 7 | `gamma` | 0.3–2.5 (useful 0.6–1.6) | `1.0` | midtone lift/crush |

Existing rows back-filled with the defaults above → **behavior unchanged until
tuned.** `weight`/`is_base` semantics untouched.

## `dissman.py` / `prompt_store.py` wiring

- `PromptStore.choose()` returns the print settings alongside the prompt
  (`PromptChoice(name, prompt, settings)`; existing callers updated).
- Settings stashed on `app.image_job` at generation time.
- `DisplayScreen.print_image_and_text` calls
  `print_pipeline.render_for_thermal(image, settings)` instead of the hardcoded
  resize/contrast block; passes the resulting 1-bit image to `p.image()`.
- `PromptStore` fallback path supplies default `PrintSettings` so a missing/
  malformed CSV still prints.

## Pi execution

- All files are pure Python on already-pinned deps (PIL, openai, escpos) →
  deploy by git-pull on reboot, no SSH-only install.
- **Printer contention:** `dissman.service` holds the printer USB. A tuning
  session (during an SSH window): `sudo systemctl stop dissman` → run probe +
  print sweeps → edit CSV → `git commit` → `sudo systemctl start dissman` (or
  reboot). Both stages run in the one SSH session so the cache never moves
  between machines.
- `--test prompt` needs `OPENAI_API_KEY` in the shell (the key in the systemd
  unit is not exported to an interactive shell); document sourcing `.env`.

## Testing

- `tests/test_print_pipeline.py` — pure-PIL unit tests for `render_for_thermal`
  (output is mode "1"; threshold cutoff splits a gray ramp correctly; dither
  produces only 0/255; resize honored; defaults preserve today's behavior).
- `PrintSettings.from_row` validation/defaults tested.
- Lab CLI and the `dissman.py` print path remain manual-verify (UI/hardware),
  per existing project convention.

## Out of scope (YAGNI)

- Automated print-quality scoring (subjective; eyeball on real paper).
- Sweep/grid files or cartesian combos (one row per prompt; edit + reprint).
- Tool writing winners back to the CSV (manual edit only).
- Multiple test input images (single `test_photo.png`; sidecar records which
  input was used so a swap is detectable).
