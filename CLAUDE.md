# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Dissman ("InsultMaster 3.0") is a coin-operated Raspberry Pi kiosk that generates AI-powered insults with doodle-style artwork. Users insert a coin, choose an insult category, get photographed, and receive a printed AI-generated caricature with a personalized insult via thermal printer.

## Commands

```
python -m pip install -r requirements.txt   # install deps (dev or Pi)
python dissman.py                            # run the kiosk app
pytest                                       # run the test suite (tests/)
pytest tests/test_insult_store.py            # run one test file
pytest tests/test_insult_store.py::test_load_preserves_duplicates   # one test
python scripts/seed_active_csvs.py           # (re)seed insults/active/*.csv
python print.py                              # thermal-printer smoke test (Pi only; prints "Hello from the Pi!" and cuts)
```

- Tests cover the pure logic only — `insult_store.py`, `prompt_store.py`, `github_sync.py`. `dissman.py` (Kivy/GPIO/camera/printer) has no automated tests; verify UI changes by running the app.
- There is no linter or build step configured.
- On a dev machine without a camera, `CameraScreen` will fail to open `cv2.VideoCapture(0)`; the rest of the flow can still be exercised.

## Running & Deploying

**On Raspberry Pi (production):**
- App lives at `/home/dissman/Documents/app/` on the Pi (hostname: `dissman`)
- Runs as a systemd service: `sudo systemctl restart dissman.service`
- Service file: `/etc/systemd/system/dissman.service`
- Starts after `graphical.target` with 10-second delay, `DISPLAY=:0`
- `OPENAI_API_KEY` is set in the service file's `Environment=` directive and in `~/.bashrc`
- Pi runs Raspberry Pi OS with labwc (Wayland) compositor via lightdm
- SSH accessible via MCP tool `claude-ssh-dissman` (an `exec` tool that runs shell commands on the Pi). Setup/replication steps are in `docs/ssh-mcp-setup.md`. The MCP server is registered at user scope and loads at **session start** — if you just registered it, restart Claude Code before its tools appear in-session.

**On desktop (development):**
- GPIO and printer gracefully disabled when not available (init wrapped in try/except)
- Coin insertion simulated with 5-second timer
- TTS uses platform-native speech (PowerShell on Windows, `say` on macOS)
- **OpenAI API is bypassed on Windows by default** (`USE_REAL_API` in `dissman.py`): the captured photo is copied through as the "doodle" so you can iterate without burning API calls. Set `DISSMAN_USE_API=1` to force the real `gpt-image-1` call. On Linux/macOS the real call always runs.
- Requires `.env` file with `OPENAI_API_KEY` (loaded via `python-dotenv`) whenever the real API is used

**Dependencies:** All versions pinned in `requirements.txt`.

Install (dev or Pi):
```
python -m pip install -r requirements.txt
```

System packages on Pi: `espeak-ng` (`sudo apt install -y espeak-ng`).

Bump procedure: install the new version of a single package locally, walk Dissman end-to-end, then update the `==` line in `requirements.txt`. The `pywin32` / `pypiwin32` / `comtypes` entries are Windows-only TTS shims that pip just skips on Linux. `gpiozero` and `python-escpos` install fine everywhere but only function on the Pi.

## Architecture

Kivy app (`dissman.py`) with screens managed by `ScreenManager`. The image-generation API call now runs in parallel with insult selection: the API kicks off as soon as the photo is captured, and the user picks their insult while the doodle is generating.

```
SplashScreen → CategoryScreen → CameraScreen → InsultScreen → (LoadScreen if not ready) → DisplayScreen
                                                                                         ↓
                                          TeachCategoryScreen ← TeachAdjScreen ← TeachNounScreen ← TeachSubmitScreen → Splash
```

- **SplashScreen**: Idle. Coin animation. GPIO pin 17. Random insults on the hour/half-hour.
- **CategoryScreen**: G-rated, R-rated, Old-timey, Anything Goes. Stores a category CODE (`g`/`r`/`old`/`all`) on the manager.
- **CameraScreen**: Runs a 30fps live preview, captures a still ~7s after entering. On capture it fires `start_image_generation` in a background thread (writes to `app.image_job`), then transitions to InsultScreen 1.5s later.
- **InsultScreen**: Generates 3 adj+noun combos from `InsultStore` for the chosen category. After pick: if `image_job.ready`, jumps to DisplayScreen; otherwise LoadScreen.
- **LoadScreen**: 16-frame "thinking" animation polling `app.image_job`. Transitions to DisplayScreen when ready; back to SplashScreen on `image_job.error`.
- **DisplayScreen**: Shows doodle + insult with staggered TTS. Thermal-prints. Two buttons: QR code, and "Insult Dissman to teach him new insults". Auto-returns to Splash after 10s (timer cancelled if user enters teach flow).
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

## Key Integration Points

- **OpenAI API**: `gpt-image-1` model via `images.edit()` endpoint. Sends captured photo with a prompt to generate an unflattering doodle. Returns base64 PNG (not URL). The prompt is chosen at call time by `PromptStore.choose()` (weighted-random over `prompts/drawing_prompts.csv`; falls back to `is_base=true` row, then a hardcoded `FALLBACK_PROMPT`). See the bypass note under "On desktop (development)".
- **GPIO**: Pin 17, coin acceptor via `gpiozero.Button`. Callback triggers screen transition.
- **Thermal Printer**: USB POS-5890 (vendor 0x0416, product 0x5011) via `python-escpos`. Images resized to 380x380, contrast reduced to 0.3 for thermal printing.
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

**Note:** `start.sh` **is now tracked in this repo** (committed verbatim from the Pi). Because the Pi runs `git reset --hard origin/main` on every boot, edits to `start.sh` here now deploy automatically on the next reboot — it is no longer Pi-only/read-only. Keep it in sync with the running copy; a malformed `start.sh` (e.g. CRLF line endings breaking the shebang) would stop the app from launching, so preserve LF endings. The systemd unit (`/etc/systemd/system/dissman.service`) is **still not tracked** — it lives only on the Pi and editing it requires SSH.
