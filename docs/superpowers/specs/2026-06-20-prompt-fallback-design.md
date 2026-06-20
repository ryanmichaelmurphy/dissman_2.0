# Prompt Fallback Chain — Design

**Date:** 2026-06-20
**Status:** Approved (pending written-spec review)
**Scope:** Production `dissman.py` image generation only. The separate
prompt-testing lab (`prompt_testing/lab.py`) is out of scope and does not share
this code.

## Revision note (folded in after the print-tuning suite landed)

This spec was first drafted before the **print-tuning & prompt-probing suite**
merged to `main`. That suite changed two assumptions, now folded in throughout:

1. **`PromptStore.choose()` returns a `PromptChoice(name, prompt, settings)`
   dataclass, not a `(name, prompt)` tuple.** Each prompt carries per-prompt
   thermal `PrintSettings` (from `print_pipeline.py`). The fallback chain must
   therefore carry `settings` end-to-end so the winning tier's tuned print
   settings still reach the printer via `job.settings`.
2. **`.gitignore` already exists.** Persisting last-successful state is now just
   adding one line to it, not creating the file.

## Problem

Every production image-generation call is currently rejected by OpenAI with
`400 moderation_blocked` (`safety_violations=[abuse]`) because the active
drawing prompt is abuse-coded. A single rejected prompt takes down the whole
kiosk experience: the user inserts a coin and gets nothing but a bounce back to
the splash screen.

We want resilience: when the chosen prompt is rejected, automatically try a
sequence of alternative prompts before giving up.

## Requirements

When the primary prompt is **rejected by OpenAI**, try, in order:

1. The `is_base == true` prompt from the prompts CSV.
2. The last prompt that previously succeeded (was not rejected by OpenAI).
3. A hardcoded final fallback: `"Make this person look really really happy and enthusiastic"`.

Only OpenAI **content rejections** advance the chain. Connectivity errors,
timeouts, 5xx, and 403 (org verification) do not — a different prompt cannot fix
those, so we stop and fail.

The chosen prompt's `PrintSettings` must follow it through to the printer,
exactly as production already does for the primary pick (`job.settings`).

## Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Where does it live? | Production `dissman.py` `_work()` only. |
| What advances the chain? | Only OpenAI content rejections (`openai.BadRequestError`, HTTP 400, e.g. `moderation_blocked`). Any other exception stops immediately. |
| Persistence of "last successful"? | On-disk, git-ignored JSON, atomic write, survives Pi reboots. Includes the prompt's print settings. |
| Code structure? | Pure helper module (`prompt_fallback.py`) + thin retry loop in `_work()`. Pure logic gets unit tests. |
| Final fallback spelling? | "enthusiastic" (corrected). |
| Record fallback as last-successful? | No. The hardcoded final fallback is never recorded. Primary / is_base / last_successful are. |
| Print settings for each tier? | is_base → its CSV row's settings; last_successful → persisted settings; final fallback → `PrintSettings.defaults()`. The winning tier's settings are written to `job.settings`. |

## Architecture

The unit of work everywhere is the existing `PromptChoice(name, prompt, settings)`
dataclass (`prompt_store.py`). The fallback module operates on `PromptChoice`
objects so settings travel with every tier.

### New module: `prompt_fallback.py` (pure, unit-tested)

```
from prompt_store import PromptChoice          # name, prompt, settings
from print_pipeline import PrintSettings       # for reconstructing persisted settings

FINAL_FALLBACK = "Make this person look really really happy and enthusiastic"
FINAL_FALLBACK_NAME = "fallback_happy"

build_candidates(primary, is_base, last_successful, final_fallback) -> list[Candidate]
read_last_successful(path) -> PromptChoice | None
write_last_successful(path, choice: PromptChoice) -> None   # atomic, best-effort
```

- A `Candidate` is `(tier, choice)` where `tier` is one of `"primary"`,
  `"is_base"`, `"last_successful"`, `"final_fallback"`, and `choice` is a
  `PromptChoice` (so `choice.name`, `choice.prompt`, `choice.settings` are all
  available downstream).
- Inputs `primary`, `is_base`, `last_successful` are each a `PromptChoice` or
  `None`. `final_fallback` is always a `PromptChoice` (never `None`).
- **`build_candidates`** assembles the four tiers in order, then:
  - drops any entry that is `None` or whose `prompt` is empty/whitespace;
  - de-dups by **prompt text** — if a tier's prompt text already appeared in an
    earlier kept tier, it is skipped (e.g. when the primary pick *is* the
    is_base prompt, or last_successful equals is_base). The earlier kept tier
    (and *its* settings) wins.
  - The `final_fallback` is the guaranteed last resort; it is retained unless its
    text duplicates an earlier kept tier, in which case the earlier one already
    carries the same prompt and the duplicate final tier is dropped.
- **`read_last_successful`** returns `None` for a missing file or any parse/IO
  error (never raises). On a valid file it returns a `PromptChoice`,
  reconstructing `settings` via `PrintSettings(**data["settings"])`; if the
  `settings` block is missing or malformed, it falls back to
  `PrintSettings.defaults()` rather than discarding the whole entry (the prompt
  text is what matters most). Expected JSON shape under "Persistence" below.
- **`write_last_successful`** serialises the `PromptChoice` (name, prompt,
  `dataclasses.asdict(choice.settings)`, ts) to a temp file in the same
  directory and `os.replace`s it into place (atomic; safe against the Pi's hard
  power-offs). Any failure is logged and swallowed — recording state must never
  break image generation.

### Addition to `prompt_store.py` (pure, testable)

```
PromptStore.base() -> PromptChoice | None
```

Returns the `is_base == true` row as a `PromptChoice` (via the existing
`_to_choice`, so its print settings are populated), or `None` if there is no such
row. Today the base prompt is only reachable indirectly inside `choose()`; the
fallback chain needs it explicitly with settings attached.

### `dissman.py` `_work()` — thin retry loop

```python
from prompt_fallback import (
    build_candidates, read_last_successful, write_last_successful,
    FINAL_FALLBACK, FINAL_FALLBACK_NAME,
)
from prompt_store import PromptChoice
from print_pipeline import PrintSettings
import openai

final_choice = PromptChoice(FINAL_FALLBACK_NAME, FINAL_FALLBACK, PrintSettings.defaults())
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
        job.settings = choice.settings           # winning tier's print settings
        job.ready = True
        if tier != "final_fallback":
            write_last_successful(LAST_SUCCESS_PATH, choice)
        print(f"[image-gen] success tier={tier} name='{choice.name}'", flush=True)
        return
    except openai.BadRequestError as e:          # content/moderation → next tier
        detail = getattr(e, "body", None) or str(e)
        print(f"[image-gen] rejected tier={tier} name='{choice.name}' "
              f"(status={getattr(e, 'status_code', None)}): {detail}", flush=True)
        continue
    except Exception as e:                        # network/5xx/403/timeout → stop
        detail = getattr(e, "body", None) or str(e)
        print(f"[image-gen] failed (non-content) tier={tier} name='{choice.name}': "
              f"{type(e).__name__}: {detail}", flush=True)
        job.error = True
        return

# every candidate was content-rejected
print("[image-gen] all candidates rejected; giving up", flush=True)
job.error = True
```

`LAST_SUCCESS_PATH = BASE_DIR / "prompts" / ".last_successful.json"`.

`job.settings` is set only on success (to the winning tier's settings), so
`DisplayScreen.print_image_and_text` — which already reads
`app.image_job.settings or PrintSettings.defaults()` — prints with the tuned
settings of whichever prompt actually produced the image. The existing
`USE_REAL_API` bypass path (Windows dev) is unchanged.

## Data flow

```
photo captured
  -> PROMPT_STORE.choose()  (primary PromptChoice, weighted random)
  -> build_candidates(primary, base, last_successful, final_fallback)  (ordered, deduped)
  -> for each (tier, choice): images.edit(prompt=choice.prompt)
       success  -> write image; job.settings = choice.settings;
                   record last_successful (unless final_fallback); job.ready
       400      -> log; try next candidate
       other    -> log; job.error; stop
  -> all rejected -> job.error
  -> DisplayScreen prints using job.settings (winning tier's PrintSettings)
```

## Persistence

- File: `prompts/.last_successful.json`
- Git-ignored. The existing **`.gitignore`** gains one line:
  ```
  prompts/.last_successful.json
  ```
  A tracked file would be overwritten by the Pi's `git reset --hard origin/main`
  on every boot, defeating the purpose. Git-ignored local state survives, exactly
  like `boot-update.log`.
- Schema (settings mirror the `PrintSettings` dataclass fields):
  ```json
  {
    "name": "<prompt name>",
    "prompt": "<text>",
    "settings": {
      "binarization": "dither", "threshold": 128, "contrast": 0.3,
      "brightness": 1.0, "resize_width": 380, "sharpness": 1.0, "gamma": 1.0
    },
    "ts": "<ISO8601 or epoch int>"
  }
  ```
  `ts` is informational (debugging). `prompt` is the field that matters most;
  `settings` lets the recorded prompt reprint with its tuned values, and degrades
  to `PrintSettings.defaults()` if absent/corrupt.

## Error handling

| Situation | Behavior |
|-----------|----------|
| `.last_successful.json` missing | `read_last_successful` → `None`; tier skipped. |
| `.last_successful.json` corrupt | `read_last_successful` → `None`; tier skipped (no crash). |
| `settings` block missing/invalid in an otherwise-valid file | settings → `PrintSettings.defaults()`; entry still usable. |
| `write_last_successful` fails (disk/IO) | Logged, swallowed; image still shown. |
| Primary == is_base (same text) | De-duped; chain still tries last_successful then final. |
| No `is_base` row in CSV | is_base tier absent; chain continues with remaining tiers. |
| All tiers content-rejected | `job.error = True` → LoadScreen routes back to Splash (existing behavior). |
| Non-content error (network/5xx/403) | `job.error = True`, stop immediately; no further API calls. |

## Testing

`tests/test_prompt_fallback.py` (pure logic):

- `build_candidates`: full four-tier ordering; dedup when primary == is_base
  (earlier tier and its settings win); dedup when last_successful == is_base;
  skips `None` and empty-string prompts; final fallback present unless a
  duplicate of an earlier kept tier; returned candidates carry the correct
  `PromptChoice` (settings included).
- `read_last_successful`: missing file → `None`; corrupt/invalid JSON → `None`;
  valid file → correct `PromptChoice`; valid file with missing/garbage
  `settings` → `PromptChoice` with `PrintSettings.defaults()`.
- `write_last_successful` → `read_last_successful` round-trip preserves name,
  prompt, and all `PrintSettings` fields; temp file is not left behind;
  overwriting an existing file works.
- `PromptStore.base()`: returns the is_base row as a `PromptChoice` with parsed
  settings; `None` when absent; tolerates a missing CSV.

The `_work()` retry loop is **not** unit-tested (it performs real API calls and
lives in the untested Kivy module, consistent with repo convention). It is
verified via the prompt-testing lab and/or a live run on the Pi, where the new
logging makes each tier attempt and rejection visible in the journal.

## Cost

Worst case (every prompt rejected) is up to 4 sequential `images.edit` calls per
coin insert. Each attempt and its outcome is logged, so the behavior is
observable. This is the intended trade-off for resilience.

## Out of scope (flagged)

- `prompt_store.FALLBACK_PROMPT` is still the moderation-blocked "middle school
  bully" prompt. With this chain it merely becomes a primary that gets rejected
  and falls through, so it is no longer fatal — but it should be replaced with a
  safe prompt in separate work.
- Rewriting the active drawing prompts in `prompts/drawing_prompts.csv` to pass
  moderation (being done via the new `prompt_testing/lab.py`).
- The prompt-testing lab itself.
```
