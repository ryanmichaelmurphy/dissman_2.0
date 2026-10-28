# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Dissman ("InsultMaster 3.0") is a coin-operated Raspberry Pi kiosk that generates AI-powered insults with doodle-style artwork. Users insert a coin, choose an insult category, get photographed, and receive a printed AI-generated caricature with a personalized insult via thermal printer.

## Companion docs

- `README.md` — public functionality / setup / data-curation docs (tracked). Update it when behavior visible to a user or operator changes.
- `prompt_testing/README.md` — the print/prompt lab CLI usage and flags reference (tracked).
- `ARTIST.md` — the artist statement, printed-QR target. **Untracked in this repo** and verbatim from the legacy `ryanmichaelmurphy/dissman` repo; don't edit the prose without the maintainer.

Keep architecture/internals detail here in `CLAUDE.md`, not duplicated into the READMEs.

## Commands

```
python -m pip install -r requirements.txt       # full app deps (dev or Pi)
python -m pip install -r requirements-dev.txt   # lab/test subset only (no Kivy/OpenCV/Pi libs)
python dissman.py                            # run the kiosk app
pytest                                       # run the test suite (tests/)
pytest tests/test_insult_store.py            # run one test file
pytest tests/test_insult_store.py::test_load_preserves_duplicates   # one test
python scripts/seed_active_csvs.py           # (re)seed insults/active/*.csv
python print.py                              # thermal-printer smoke test (Pi only; prints "Hello from the Pi!" and cuts)
```

- Tests cover the pure logic only — `insult_store.py`, `prompt_store.py`, `print_pipeline.py`, `prompt_fallback.py`, `image_models.py`, `github_sync.py`, and the lab helpers (`prompt_testing/lab.py`). `dissman.py` (Kivy/GPIO/camera/printer) has no automated tests; verify UI changes by running the app.
- **Windows dev env:** the bare `python`/`pytest` above assume the deps import. On the maintainer's Windows box the *system* Python lacks Pillow/openai — the working env is conda (Kivy 2.3.1 only ships wheels through Python 3.12). Run the app and tests through it: `& "C:\Users\Ryan\conda-envs\dissman\python.exe" -m pytest`. `pytest` is not pinned in `requirements.txt`; if a fresh env reports `No module named 'pytest'`, `pip install` it into that conda env rather than falling back to system Python.
- There is no linter or build step configured.
- On a dev machine without a camera, `CameraScreen` will fail to open `cv2.VideoCapture(0)`; the rest of the flow can still be exercised.

## Git Workflow

- Commit every change with a descriptive message.
- On a branch other than `main`: push after committing.
- On `main`: commit, but **do not push until the maintainer explicitly says to** (pushes to `main` deploy to the Pi on its next power-cycle).

## Running & Deploying

**On Raspberry Pi (production):**
- App lives at `/home/dissman/Documents/app/` on the Pi (hostname: `dissman`)
- Runs as a systemd service: `sudo systemctl restart dissman.service`
- Service file: `/etc/systemd/system/dissman.service`
- Starts after `graphical.target` with 10-second delay, `DISPLAY=:0`
- `OPENAI_API_KEY` is set in the service file's `Environment=` directive and in `~/.bashrc`
- Pi runs Raspberry Pi OS with labwc (Wayland) compositor via lightdm
- SSH accessible via MCP tool `claude-ssh-dissman` (an `exec` tool that runs shell commands on the Pi). Setup/replication steps are in `docs/ssh-mcp-setup.md`. The server can be registered two ways (same name, don't run both): a committed project-scoped `.mcp.json` at the repo root that reads per-device env vars (`DISSMAN_SSH_MCP_PATH`, `DISSMAN_PI_PASSWORD`), or a per-machine user-scope entry. Either way the MCP tools load at **session start** — if you just registered it, restart Claude Code before its tools appear in-session. The `.mcp.json` never holds the password; it stays in a per-device env var, out of git.

**On desktop (development):**
- GPIO and printer gracefully disabled when not available (init wrapped in try/except)
- Coin insertion simulated with 5-second timer
- TTS uses platform-native speech (PowerShell on Windows, `say` on macOS)
- **OpenAI API is bypassed on Windows by default** (`USE_REAL_API` in `dissman.py`): the captured photo is copied through as the "doodle" so you can iterate without burning API calls. Set `DISSMAN_USE_API=1` to force the real image API call. On Linux/macOS the real call always runs.
- Requires `.env` file with `OPENAI_API_KEY` (loaded via `python-dotenv`) whenever the real API is used

**Dependencies:** All versions pinned in `requirements.txt` (full app). `requirements-dev.txt` is a **subset** of the same pins — just what the lab (`prompt_testing/lab.py`) and the pytest suite need (`openai`, `python-dotenv`, `pillow`, `pytest` + transitives), with no Kivy/OpenCV/numpy/Pi hardware libs. Use a virtualenv so the pins are authoritative and don't leak into the global interpreter.

Install (dev or Pi):
```
python -m venv .venv                            # then activate: .venv\Scripts\Activate.ps1 (Win) or source .venv/bin/activate
python -m pip install -r requirements.txt       # full app (running dissman.py)
python -m pip install -r requirements-dev.txt   # lab/test only (prompt tuning, pytest)
```

System packages on Pi: `espeak-ng` (`sudo apt install -y espeak-ng`).

Bump procedure: install the new version of a single package locally, walk Dissman end-to-end, then update the `==` line in `requirements.txt`. **If the bumped package is also in `requirements-dev.txt` (`openai`, `python-dotenv`, `pillow`, and their shared transitives), mirror the new pin there too** so the two files stay in lockstep. The `pywin32` / `pypiwin32` / `comtypes` entries are Windows-only TTS shims that pip just skips on Linux. `gpiozero` and `python-escpos` install fine everywhere but only function on the Pi.

## Architecture

Kivy app (`dissman.py`) with screens managed by `ScreenManager`. The image-generation API call now runs in parallel with insult selection: the API kicks off as soon as the photo is captured, and the user picks their insult while the doodle is generating.

```
SplashScreen → CategoryScreen → CameraScreen → InterludeScreen → InsultScreen → (LoadScreen if not ready) → DisplayScreen
                                                                                                           ↓
                                                            TeachCategoryScreen ← TeachAdjScreen ← TeachNounScreen ← TeachSubmitScreen → Splash
```

- **SplashScreen**: Idle. Coin animation. GPIO pin 17. Random insults on the hour/half-hour.
- **CategoryScreen**: G-rated, R-rated, Old-timey, Anything Goes. Stores a category CODE (`g`/`r`/`old`/`all`) on the manager.
- **CameraScreen**: Runs a 30fps live preview, captures a still ~7s after entering. On capture it replaces `app.image_job` with a **fresh `ImageJob`** (so a previous customer's slow thread can only write to its own orphaned job) and fires `start_image_generation` in a background thread, then slides to InterludeScreen 1.5s later. A failed frame read retries in 0.5s.
- **InterludeScreen**: 16-frame "thinking" animation (~4.8s), then InsultScreen.
- **InsultScreen**: Generates 3 adj+noun combos from `InsultStore` for the chosen category. After pick: if `image_job.ready`, jumps to DisplayScreen; otherwise LoadScreen.
- **LoadScreen**: 16-frame "thinking" animation polling `app.image_job`. Transitions to DisplayScreen when ready. After 120s without a result it degrades the job to the raw photo (see OpenAI API below). `image_job.error` (only set when there's no photo at all) logs, speaks "Something broke", and returns to Splash.
- **DisplayScreen**: Shows doodle + insult with staggered TTS (when `image_job.degraded`, it says "My art brain is broken. This is what you actually look like." and shifts the insult speech later). Thermal-prints. Two buttons: QR code, and "Insult Dissman to teach him new insults". Auto-returns to Splash after 10s (timer cancelled if user enters teach flow).
- **TeachCategoryScreen → TeachAdjScreen → TeachNounScreen → TeachSubmitScreen**: User-submitted insult flow. Adjective and noun entered via Kivy `VKeyboard`. Submission appended to `insults/submissions/{adjectives,nouns}.csv` and pushed to GitHub asynchronously via `github_sync.push_submission_async`.

## Insult storage

Word lists live in CSV files under `insults/`:

```
insults/
├── active/
│   ├── adjectives.csv      # word,category,added_date  — used by InsultStore
│   └── nouns.csv           # word,category,added_date
└── submissions/
    ├── adjectives.csv      # word,category,submission_date  — user submissions
    └── nouns.csv
```

- `category` is one of `g`, `r`, `old`. `"all"` is a runtime union across the three.
- **Duplicates in active CSVs are intentional and preserved.** A word listed N times is N times more likely to be drawn (via `random.choice`).
- To approve a user submission: copy the row from `insults/submissions/*.csv` into the matching `insults/active/*.csv` (optionally multiple times to boost frequency). The Pi will pick it up on the next boot via `start.sh`'s auto-pull.
- `insult_store.py` owns reads (`InsultStore.adjectives(cat)`, `.nouns(cat)`) and writes (`InsultStore.record_submission(cat, pos, word)`).
- `github_sync.py` owns the async `git add/commit/push` of submissions. Network failures are logged but never raised.

### Drawing prompts & per-prompt print settings

- `prompts/drawing_prompts.csv` columns: `name,weight,is_base,binarization,threshold,contrast,brightness,resize_width,sharpness,gamma,prompt`. The print-variable columns (ordered most→least impactful) tune thermal output per prompt; defaults `dither,128,0.3,1.0,380,1.0,1.0` preserve the original hardcoded behavior.
- `prompt_store.py` `PromptStore.choose()` returns a `PromptChoice(name, prompt, settings)` where `settings` is a `PrintSettings` parsed from the row.
- `print_pipeline.py` `render_for_thermal(image, settings)` is the **single source of truth** for thermal rendering (resize → contrast → brightness → sharpness → gamma → binarize → mode "1"). Imported by **both** `dissman.py`'s print path and `prompt_testing/lab.py`, so tuned == printed.
- **Tuning loop** (`prompt_testing/lab.py`, run during an SSH window): `dissman.service` owns the printer USB, so `sudo systemctl stop dissman` first. `--test prompt` calls the API on `test_photo.png`, caches the image + sidecar (auto-invalidates when the prompt text changes), and prints the moderation `.body` on rejection — use it to iterate wording past `moderation_blocked`. `--test print` reprints the cached image through the row's settings (`--preview` writes a PNG instead). Edit the CSV row → reprint → repeat; record winners by hand; `git commit`; `sudo systemctl start dissman`. The `cache/`/`preview/` dirs are gitignored.

## Key Integration Points

- **OpenAI API**: `images.edit()` with `quality="low"`, `size="1024x1024"` (output is shrunk to ~380px and binarized, so low is indistinguishable on paper). Sends captured photo with a prompt to generate an unflattering doodle. Returns base64 PNG (not URL). `image_models.py` is the **single source of truth** for the call (`edit_image`, `EDIT_PARAMS`, `make_client` with `timeout=90, max_retries=1`) — imported by both `dissman.py` and `prompt_testing/lab.py`.
  - **Models** come from the tracked `prompts/image_models.txt` (preference order; falls back to `image_models.DEFAULT_MODELS` if missing). `gpt-image-1` was retired 2026-10-23; the current list is `gpt-image-2.5-flare` → `gpt-image-2.5-sunburst` → `gpt-image-1-mini` (retires 2026-12-01). When OpenAI retires a model, edit that file and push — no code change.
  - **Error classes** (`image_models.classify_error`): **content** = `BadRequestError` with `moderation_blocked`/safety → next *prompt*; **model** = `NotFoundError`, `PermissionDeniedError`, or any other 400 → next *model*, and `ModelRoster` drops it until reboot (last-good model is tried first); **fatal** = auth, rate limit/quota, connection/timeout, 5xx, anything else → stop.
  - **Degraded print**: when the chain is exhausted, hits a fatal error, exceeds `CHAIN_DEADLINE_SECONDS` (150s), or there's no key, `degrade_to_photo` makes the raw photo the "doodle" (`job.degraded=True`, default print settings) so the customer still gets a printout. The client is built lazily, so a missing key degrades instead of crashing at import.
  - **Logging**: every failure logs `describe_error` (type, status, request_id, code, body, `__cause__`); non-API exceptions add a traceback; success logs model + `usage`. At boot `log_image_api_status()` logs key presence (last 4 chars), openai version and the model list, then probes each model with `models.retrieve` and logs `MODEL UNAVAILABLE: <model>` for a 404 — grep for it on the `pi-logs` branch.
  - **Prompt fallback chain** (outer loop; models are the inner loop): on content rejection, retries through **primary** (chosen at call time by `PromptStore.choose()`) → **is_base** (`PromptStore.base()`) → **last_successful** (recovered from git-ignored `prompts/.last_successful.json`, which atomically records the last prompt that passed moderation and survives reboots) → **final fallback** (`"Make this person look really really happy and enthusiastic"`). Each prompt's print settings travel with it through the chain. The final fallback is never persisted. The chain logic lives in `prompt_fallback.py` + `image_models.py` (pure, unit-tested) plus the `_work()` retry loop in `dissman.py`. See the bypass note under "On desktop (development)".
- **GPIO**: Pin 17, coin acceptor via `gpiozero.Button`. Callback triggers screen transition.
- **Thermal Printer**: USB POS-5890 (vendor 0x0416, product 0x5011) via `python-escpos`. Image preprocessing (resize/contrast/binarization) is per-prompt via `print_pipeline.render_for_thermal`; see "Drawing prompts & per-prompt print settings".
- **TTS**: `speak()` function dispatches to `espeak-ng` (Linux), `say` (macOS), or PowerShell (Windows) via subprocess.

## UI

- Display: 800x480 fullscreen borderless (Pi 7-inch touchscreen)
- Layout defined in `insultmaster3.kv`
- Theme: lime green text (#C9D32D) on black buttons, orange accents, dark blue background
- Font: FreeMono throughout

## Pi Access Constraint (IMPORTANT)

**The maintainer does not have regular SSH or physical access to the Pi.** The only opportunity for changes to land in production is when a third party power-cycles the device remotely. That reboot triggers `start.sh`, which pulls `origin/main` and launches the app.

**Implications for any proposed change:**
- Anything that lands *only* by `git pull` (Python code, kv files, CSV data, image assets already tracked in the repo) deploys automatically on the next reboot. Safe.
- Anything that requires SSH (pip installs, apt installs, systemd unit edits, file copies outside the repo, GPIO/wiring tweaks, `git config` changes, credential rotation, etc.) **cannot be assumed to happen.** Flag these prominently in the PR/commit/plan and propose a workaround whenever possible:
  - Prefer pure-stdlib code over new pip deps.
  - If a new pip dep is unavoidable, consider vendoring it into the repo (`vendor/<pkg>/`) and adding the path to `sys.path` in `start.sh` (now tracked — see below — so such edits deploy via `git pull`).
  - If config must change, fold it into `start.sh` (now tracked in the repo, so edits deploy on the next pull) rather than the systemd unit (Pi-only, requires SSH).
  - If credentials must rotate, document the exact one-line command the operator would run.
- When in doubt, ask before assuming Pi-side work will get done.

## Deploying Changes to Pi

The Pi's app directory is a git repo tracking `origin/main`. On every service start, `start.sh` runs `git fetch && git reset --hard origin/main` (with a 20s timeout, offline-tolerant) before launching `python3 dissman.py`. To deploy: push to `main`, then on the Pi:

```bash
sudo systemctl restart dissman.service
sudo journalctl -u dissman.service -b --no-pager -f
tail /home/dissman/Documents/app/boot-update.log
```

`start.sh`, `boot-update.log`, and the systemd unit (`/etc/systemd/system/dissman.service` with `ExecStart=/home/dissman/Documents/app/start.sh`) handle the auto-pull. Old non-tracked files (`.bak`, `.save`, etc.) still live in the Pi's app dir but are ignored.

**Reading Pi logs remotely (no SSH):** `start.sh` also publishes logs to the orphan `pi-logs` branch on GitHub. It keeps a git worktree at `/home/dissman/Documents/app-logs` checked out on `pi-logs`; the app launches under `python3 -u … | tee` so stdout/stderr are written to disk line-by-line and survive a hard power-off. On each boot, `start.sh` commits the *previous* session's captured log (plus `boot-update.log`) to `pi-logs` and pushes — so the session that just died shows up on the `pi-logs` branch after the next power-cycle. Read errors there instead of needing SSH. The worktree and branch never touch `main` or the boot-time `git reset --hard origin/main`, and the `pi-logs` push reuses the same git credentials as `github_sync`.

**Note:** `start.sh` **is now tracked in this repo** (committed verbatim from the Pi). Because the Pi runs `git reset --hard origin/main` on every boot, edits to `start.sh` here now deploy automatically on the next reboot — it is no longer Pi-only/read-only. Keep it in sync with the running copy; a malformed `start.sh` (e.g. CRLF line endings breaking the shebang) would stop the app from launching, so preserve LF endings. The systemd unit (`/etc/systemd/system/dissman.service`) is **still not tracked** — it lives only on the Pi and editing it requires SSH.
