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
