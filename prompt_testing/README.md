# Prompt & Print Testing Lab

`lab.py` lets you test a drawing prompt in two ways, **without running the kiosk app**:

1. **Prompt test** — does this prompt pass OpenAI's moderation, and what image does it produce?
2. **Print test** — how does that image look on the thermal printer with a given set of print settings?

Each prompt lives as one row in [`../prompts/drawing_prompts.csv`](../prompts/drawing_prompts.csv). You point the lab at a row **by name**, choose what to test, look at the result, tweak the row, and repeat.

---

## Quick start

```bash
# 1. Generate an image for a prompt (and check it isn't moderation-blocked)
python3 prompt_testing/lab.py --name middle_school_bully --test prompt

# 2. Print that cached image using the row's current print settings
python3 prompt_testing/lab.py --name middle_school_bully --test print

# ...edit the row in prompts/drawing_prompts.csv, then reprint:
python3 prompt_testing/lab.py --name middle_school_bully --test print
```

The image is generated **once** and cached, so step 2 can be repeated as many times as you like while tuning print settings — no extra API calls or cost.

---

## Flags

| Flag | Required | Applies to | What it does |
|------|----------|-----------|--------------|
| `--name <row>` | yes | both | Which prompt to test, matched against the `name` column of the CSV (e.g. `middle_school_bully`, `goya_saturn`). |
| `--test prompt` | yes | — | Call the OpenAI image API on the test photo, cache the result, and report **accepted / rejected** (printing the exact moderation reason on rejection). |
| `--test print` | yes | — | Render the cached image through the row's print settings and **print it** on the POS-5890. |
| `--preview` | no | `--test print` | Write the processed 1-bit image to `preview/<name>.png` **instead of printing** (no paper, works off the Pi). |
| `--force` | no | `--test prompt` | Regenerate the image even if a fresh one is cached. |
| `--input <path>` | no | `--test prompt` | Use a different source photo (default: `prompt_testing/test_photo.png`). |
| `--csv <path>` | no | both | Use a different prompts CSV (default: `prompts/drawing_prompts.csv`). |

If `--name` doesn't match any row, the lab lists the available names and exits.

---

## Testing a particular **prompt** (moderation + generation)

```bash
python3 prompt_testing/lab.py --name middle_school_bully --test prompt
```

- **Accepted** → the raw image is cached to `cache/<name>.png`.
- **Rejected** → no image; the exact API reason (e.g. `moderation_blocked`, `safety_violations=[abuse]`) is printed and saved to `cache/<name>.json`.

To get a blocked prompt to pass: edit the `prompt` text in the CSV row, then run `--test prompt` again. The lab notices the wording changed and **regenerates automatically**. Use `--force` to regenerate even when the wording is unchanged.

---

## Testing **print settings** (tuning output quality)

```bash
python3 prompt_testing/lab.py --name middle_school_bully --test print
```

This reads the print-variable columns from the prompt's CSV row and applies them before printing. The printed header shows the exact settings used, so you can label paper output. To iterate: **edit the row → reprint → compare**.

Preview to screen instead of wasting paper (also runs off the Pi):

```bash
python3 prompt_testing/lab.py --name middle_school_bully --test print --preview
# writes prompt_testing/preview/middle_school_bully.png
```

### Print variables (CSV columns), most → least impactful

Edit these in the prompt's row in `prompts/drawing_prompts.csv`:

| Column | Range | Default | Effect |
|--------|-------|---------|--------|
| `binarization` | `threshold` \| `dither` | `dither` | How gray becomes pure black/white. **Biggest lever.** `threshold` = sharp line-art; `dither` = shading via dot patterns. |
| `threshold` | 0–255 (useful 90–180) | `128` | Black/white cutoff. Only used when `binarization=threshold`. Lower = more white, higher = more black. |
| `contrast` | 0.0–3.0 (useful 0.5–2.0) | `0.3` | Pre-binarization contrast. `1.0` = unchanged. |
| `brightness` | 0.0–2.0 (useful 0.7–1.4) | `1.0` | Pre-binarization brightness. `1.0` = unchanged. |
| `resize_width` | 128–384 px | `380` | Print width in dots (POS-5890 ≈ 384 wide). Affects line weight/detail. |
| `sharpness` | 0.0–3.0 (useful 1.0–2.5) | `1.0` | Edge crispness. `>1.0` sharpens lines before binarizing. |
| `gamma` | 0.3–2.5 (useful 0.6–1.6) | `1.0` | Midtone lift/crush. `<1.0` brightens mids, `>1.0` darkens them. |

A good starting recipe for crisp doodle line-art: `binarization=threshold`, then nudge `threshold`, `contrast`, and `sharpness`.

---

## Running on the Pi

The kiosk service owns the printer's USB port, so **stop it first**, and make sure the API key is available for `--test prompt`:

```bash
sudo systemctl stop dissman
set -a; . /home/dissman/Documents/app/.env; set +a   # load OPENAI_API_KEY
cd /home/dissman/Documents/app

python3 prompt_testing/lab.py --name middle_school_bully --test prompt
python3 prompt_testing/lab.py --name middle_school_bully --test print
#   ...edit prompts/drawing_prompts.csv, reprint, repeat...

git commit -am "tune middle_school_bully"   # winners are recorded by hand
sudo systemctl start dissman
```

Once committed to `main`, the new prompt text and print settings reach the live kiosk on the next reboot — the same `print_pipeline` powers both the lab and the app, so what you tune here is exactly what prints in production.

---

## Notes

- `cache/` (generated images + result sidecars) and `preview/` are local-only and gitignored — they never get committed.
- Winning settings are recorded by **manually editing** the CSV. The lab never writes to your prompt data.
- `--test prompt` needs `OPENAI_API_KEY` (from `.env`). `--test print` needs the real printer (or use `--preview`).
