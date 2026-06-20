# Dissman

*"InsultMaster 3.0"* — a coin-operated Raspberry Pi kiosk that takes your photo, generates a doodle-style caricature with an AI-written insult, and prints it on a thermal receipt printer.

For the artistic intent behind the piece, see [ARTIST.md](ARTIST.md).

## How it works

1. **Insert a coin.** The splash screen waits for a signal on GPIO pin 17 (coin acceptor). While waiting, Dissman periodically mutters insults aloud.
2. **Choose a category.** G-rated, R-rated, Old-timey, or Anything Goes.
3. **Smile (or don't).** A webcam captures your photo, a white flash fires, and the image is sent to OpenAI's `gpt-image-1` to be redrawn as a doodle.
4. **Pick your insult.** While the doodle is generating, you're shown three randomly assembled adjective + noun combos from the active word list for the category you picked.
5. **Receive your insult.** The doodle and insult are displayed on screen, spoken aloud with dramatic timing, and printed on the thermal printer. A QR code is also printed linking to this repo and the artist statement.
6. **Optionally, teach Dissman a new insult.** A second flow lets users submit their own adjectives and nouns, which are pushed to GitHub for later curation.

## Hardware

- Raspberry Pi (running Raspberry Pi OS, labwc/Wayland under lightdm)
- 7-inch 800x480 touchscreen
- USB webcam
- POS-5890 USB thermal printer (vendor `0x0416`, product `0x5011`)
- Coin acceptor wired to GPIO pin 17

## Software

- **App:** Kivy 2.3.1 (`dissman.py`, layout in `insultmaster3.kv`)
- **Image generation:** OpenAI `gpt-image-1` via `images.edit()`
- **TTS:** `espeak-ng` (Linux), `say` (macOS), or PowerShell `System.Speech` (Windows)
- **Printer:** `python-escpos`
- **GPIO:** `gpiozero`

All dependencies are pinned in `requirements.txt`.

## Running locally

Kivy 2.3.1 only ships wheels through Python 3.12.

```bash
python -m pip install -r requirements.txt
python dissman.py
```

By default on Windows, the OpenAI call is bypassed (the captured photo is copied straight through) to avoid API spend during development. Set `DISSMAN_USE_API=1` to opt in to the real call. Linux/Pi always uses the real API.

You'll need an `OPENAI_API_KEY` in `.env` (loaded by `python-dotenv`).

## Tests

```bash
python -m pytest tests/ -q
```

Tests cover the pure-Python data layer (`insult_store.py`, `github_sync.py`, `prompt_store.py`). The Kivy UI isn't unit-tested — verify changes by running the app.

## Data: how to add or change insults

Active words and submitted words live in separate CSV trees so submissions can be reviewed before going live.

```
insults/
├── active/         # what the kiosk actually draws from
│   ├── adjectives.csv
│   └── nouns.csv
└── submissions/    # what users typed in via the teach flow
    ├── adjectives.csv
    └── nouns.csv

prompts/
└── drawing_prompts.csv   # weighted drawing-style prompts for the image API
```

**Categories:** `g`, `r`, `old`. (`all` is a runtime union of the three.)

**Duplicates are intentional.** A word listed N times in an active CSV is N times more likely to be drawn. Don't dedupe.

**Approving a submission:** copy the row from `insults/submissions/*.csv` into the matching `insults/active/*.csv`. Paste 2–3 times to boost its draw frequency.

**Tuning drawing prompts:** edit weights in `prompts/drawing_prompts.csv`. No code change needed. Each row also carries per-prompt print settings (`binarization,threshold,contrast,brightness,resize_width,sharpness,gamma`) that control how the doodle is rendered for the thermal printer.

**Testing prompts & print quality:** `prompt_testing/lab.py` lets you (1) check whether a prompt passes OpenAI moderation and (2) tune the print settings against the real printer, without touching the kiosk app. One API call is cached, then reprinted with different settings:

```bash
# On the Pi, during an SSH session (the kiosk owns the printer, so stop it first):
sudo systemctl stop dissman
python3 prompt_testing/lab.py --name middle_school_bully --test prompt   # generate + moderation check
python3 prompt_testing/lab.py --name middle_school_bully --test print    # print with the row's settings
#   ... edit the row in prompts/drawing_prompts.csv, reprint, repeat ...
git commit -am "tune middle_school_bully" && sudo systemctl start dissman
```

Use `--preview` to write the 1-bit image to a PNG instead of printing (works off-Pi). The rendering is shared with the live app (`print_pipeline.py`), so what you tune is exactly what prints.

## Deployment

The Pi runs the app as a systemd service (`/etc/systemd/system/dissman.service`) that launches `start.sh` after `graphical.target`. `start.sh` does a `git fetch && git reset --hard origin/main` (offline-tolerant, 20s timeout) on every boot, so anything merged to `main` lands in production on the next power-cycle.

## Repo layout

```
dissman.py             # main Kivy app
insultmaster3.kv       # screen layouts
insult_store.py        # read/write of active + submitted CSVs
prompt_store.py        # weighted-random selection of drawing prompts + settings
print_pipeline.py      # shared thermal-render pipeline (app + lab)
github_sync.py         # async push of user-submitted insults
start.sh               # boot wrapper: auto-pull then launch
insults/               # word data (active + submissions)
prompts/               # drawing prompt data
prompt_testing/        # lab.py: prompt-moderation & print-tuning CLI
tests/                 # pytest suite for the data layer
```
