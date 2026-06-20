# Prompt Fallback Chain — Design

**Date:** 2026-06-20
**Status:** Approved (pending written-spec review)
**Scope:** Production `dissman.py` image generation only. The separate
prompt-testing script is out of scope and does not share this code.

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

## Decisions (from brainstorming)

| Question | Decision |
|----------|----------|
| Where does it live? | Production `dissman.py` `_work()` only. |
| What advances the chain? | Only OpenAI content rejections (`openai.BadRequestError`, HTTP 400, e.g. `moderation_blocked`). Any other exception stops immediately. |
| Persistence of "last successful"? | On-disk, git-ignored JSON, atomic write, survives Pi reboots. |
| Code structure? | Pure helper module (`prompt_fallback.py`) + thin retry loop in `_work()`. Pure logic gets unit tests. |
| Final fallback spelling? | "enthusiastic" (corrected). |
| Record fallback as last-successful? | No. The hardcoded final fallback is never recorded. Primary / is_base / last_successful are. |

## Architecture

### New module: `prompt_fallback.py` (pure, unit-tested)

```
FINAL_FALLBACK = "Make this person look really really happy and enthusiastic"
FINAL_FALLBACK_NAME = "fallback_happy"

build_candidates(primary, is_base, last_successful, final_fallback) -> list[Candidate]
read_last_successful(path) -> tuple[name, prompt] | None
write_last_successful(path, name, prompt) -> None   # atomic, best-effort
```

- A `Candidate` is `(tier, name, prompt)` where `tier` is one of
  `"primary"`, `"is_base"`, `"last_successful"`, `"final_fallback"`.
- Each of `primary`, `is_base`, `last_successful` is a `(name, prompt)` tuple or
  `None`. `final_fallback` is always present.
- **`build_candidates`** assembles the four tiers in order, then:
  - drops any entry that is `None` or whose `prompt` is empty/whitespace;
  - de-dups by **prompt text** — if a tier's prompt text already appeared in an
    earlier kept tier, it is skipped (e.g. when the primary pick *is* the
    is_base prompt, or last_successful equals is_base).
  - The `final_fallback` is always retained (it is the guaranteed last resort);
    if its text coincidentally matches an earlier tier, the earlier one wins and
    the final tier is dropped as a duplicate — acceptable, because that earlier
    tier already carries the same prompt.
- **`read_last_successful`** returns `None` for a missing file or any parse/IO
  error (never raises). Expected JSON: `{"name": str, "prompt": str, "ts": str}`.
- **`write_last_successful`** writes to a temp file in the same directory and
  `os.replace`s it into place (atomic; safe against the Pi's hard power-offs).
  Any failure is logged and swallowed — recording state must never break image
  generation.

### Addition to `prompt_store.py` (pure, testable)

```
PromptStore.base() -> tuple[name, prompt] | None
```

Returns the `is_base == true` row as `(name, prompt)`, or `None` if there is no
such row. Today the base prompt is only reachable indirectly inside `choose()`;
the fallback chain needs it explicitly.

### `dissman.py` `_work()` — thin retry loop

```python
from prompt_fallback import (
    build_candidates, read_last_successful, write_last_successful,
    FINAL_FALLBACK, FINAL_FALLBACK_NAME,
)
import openai

candidates = build_candidates(
    primary=PROMPT_STORE.choose(),
    is_base=PROMPT_STORE.base(),
    last_successful=read_last_successful(LAST_SUCCESS_PATH),
    final_fallback=(FINAL_FALLBACK_NAME, FINAL_FALLBACK),
)

for tier, name, prompt in candidates:
    print(f"[image-gen] attempt tier={tier} name='{name}'", flush=True)
    try:
        with open(source_image_path, "rb") as f:
            response = client.images.edit(
                model="gpt-image-1", image=f, prompt=prompt, n=1, size="1024x1024",
            )
        data = base64.b64decode(response.data[0].b64_json)
        with open(out_path, "wb") as f:
            f.write(data)
        job.image_path = out_path
        job.ready = True
        if tier != "final_fallback":
            write_last_successful(LAST_SUCCESS_PATH, name, prompt)
        print(f"[image-gen] success tier={tier} name='{name}'", flush=True)
        return
    except openai.BadRequestError as e:          # content/moderation → next tier
        detail = getattr(e, "body", None) or str(e)
        print(f"[image-gen] rejected tier={tier} name='{name}' "
              f"(status={getattr(e, 'status_code', None)}): {detail}", flush=True)
        continue
    except Exception as e:                        # network/5xx/403/timeout → stop
        detail = getattr(e, "body", None) or str(e)
        print(f"[image-gen] failed (non-content) tier={tier} name='{name}': "
              f"{type(e).__name__}: {detail}", flush=True)
        job.error = True
        return

# every candidate was content-rejected
print("[image-gen] all candidates rejected; giving up", flush=True)
job.error = True
```

`LAST_SUCCESS_PATH = BASE_DIR / "prompts" / ".last_successful.json"`.

The existing `USE_REAL_API` bypass path (Windows dev) is unchanged.

## Data flow

```
photo captured
  -> PROMPT_STORE.choose()  (primary, weighted random)
  -> build_candidates(primary, base, last_successful, final_fallback)  (ordered, deduped)
  -> for each candidate: images.edit(...)
       success  -> write image; record last_successful (unless final_fallback); job.ready
       400      -> log; try next candidate
       other    -> log; job.error; stop
  -> all rejected -> job.error
```

## Persistence

- File: `prompts/.last_successful.json`
- Git-ignored. A **new `.gitignore`** is added with at least:
  ```
  prompts/.last_successful.json
  ```
  A tracked file would be overwritten by the Pi's `git reset --hard origin/main`
  on every boot, defeating the purpose. Git-ignored local state survives, exactly
  like `boot-update.log`.
- Schema: `{"name": "<prompt name>", "prompt": "<text>", "ts": "<ISO8601>"}`.
  `ts` is informational (debugging); `prompt` is the field that matters.

## Error handling

| Situation | Behavior |
|-----------|----------|
| `.last_successful.json` missing | `read_last_successful` → `None`; tier skipped. |
| `.last_successful.json` corrupt | `read_last_successful` → `None`; tier skipped (no crash). |
| `write_last_successful` fails (disk/IO) | Logged, swallowed; image still shown. |
| Primary == is_base (same text) | De-duped; chain still tries last_successful then final. |
| No `is_base` row in CSV | is_base tier absent; chain continues with remaining tiers. |
| All tiers content-rejected | `job.error = True` → LoadScreen routes back to Splash (existing behavior). |
| Non-content error (network/5xx/403) | `job.error = True`, stop immediately; no further API calls. |

## Testing

`tests/test_prompt_fallback.py` (pure logic):

- `build_candidates`: full four-tier ordering; dedup when primary == is_base;
  dedup when last_successful == is_base; skips `None` and empty-string prompts;
  final fallback always present unless duplicate of an earlier kept tier.
- `read_last_successful`: missing file → `None`; corrupt/invalid JSON → `None`;
  valid file → correct `(name, prompt)`.
- `write_last_successful` → `read_last_successful` round-trip; temp file is not
  left behind; overwriting an existing file works.
- `PromptStore.base()`: returns the is_base row; `None` when absent; tolerates a
  missing CSV.

The `_work()` retry loop is **not** unit-tested (it performs real API calls and
lives in the untested Kivy module, consistent with repo convention). It is
verified via the prompt-testing script and/or a live run on the Pi, where the new
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
  moderation (the user is doing this independently).
- The prompt-testing script itself.
```
