#!/usr/bin/env python3
"""Prompt-probing and print-tuning lab for Dissman.

Two modes against ONE prompt row of prompts/drawing_prompts.csv:

  --test prompt : call the OpenAI image edit on test_photo.png, cache the raw
                  PNG + a JSON sidecar, and report accepted / rejected (+ the
                  exact moderation .body). Auto-regenerates when the row's
                  prompt text changed since the cached run; --force always does.

  --test print  : render the cached image through that row's print variables
                  and print it on the POS-5890. --preview writes a PNG instead.

--model picks the image model (default: first line of prompts/image_models.txt);
images are cached per (prompt, model) so models can be compared side by side.
The call itself is image_models.edit_image, i.e. exactly what the kiosk sends.

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
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from print_pipeline import PrintSettings, render_for_thermal  # noqa: E402
from image_models import (  # noqa: E402
    EDIT_PARAMS, describe_error, edit_image, load_models, make_client,
)

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
PREVIEW_DIR = HERE / "preview"
DEFAULT_INPUT = HERE / "test_photo.png"
DEFAULT_CSV = ROOT / "prompts" / "drawing_prompts.csv"
MODELS_FILE = ROOT / "prompts" / "image_models.txt"

# USD per 1M tokens, for the cost readout. From developers.openai.com/api/docs/pricing
# (2026-10). Unknown models use the 2.5 rates.
TOKEN_RATES = {
    "gpt-image-2.5-flare":    {"image_in": 8.00, "text_in": 5.00, "image_out": 30.00},
    "gpt-image-2.5-sunburst": {"image_in": 8.00, "text_in": 5.00, "image_out": 30.00},
    "gpt-image-1-mini":       {"image_in": 2.50, "text_in": 2.00, "image_out": 8.00},
}
DEFAULT_RATES = TOKEN_RATES["gpt-image-2.5-flare"]

# On the Pi, dissman.service carries OPENAI_API_KEY in its Environment= line.
# Reading it from there lets the lab run without manually exporting the key.
SERVICE_UNIT = Path("/etc/systemd/system/dissman.service")


def key_from_systemd_unit(text):
    """Extract OPENAI_API_KEY from a systemd unit's Environment= line, or None."""
    m = re.search(r"OPENAI_API_KEY=(\S+)", text)
    if not m:
        return None
    return m.group(1).strip().strip('"').strip("'") or None


def resolve_api_key():
    """Find the OpenAI key without requiring a manual export.

    Order: process env (which includes anything load_dotenv put there from a
    .env) first; then, on the Pi, fall back to the dissman.service unit.
    """
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    if SERVICE_UNIT.exists():
        return key_from_systemd_unit(SERVICE_UNIT.read_text())
    return None


def load_row(rows, name):
    for r in rows:
        if r.get("name") == name:
            return r
    return None


def cache_is_fresh(sidecar, prompt_text, quality=None):
    """Fresh only if ACCEPTED and the prompt (and quality, if given) is unchanged."""
    if not sidecar:
        return False
    if sidecar.get("status") != "accepted":
        return False
    if quality is not None and sidecar.get("quality") != quality:
        return False
    return sidecar.get("prompt") == prompt_text


def usage_dict(usage):
    """SDK usage object (or dict/None) -> plain dict."""
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return dict(usage) if isinstance(usage, dict) else {}


def estimate_cost(usage, model):
    """Estimated USD for one call from its usage block, or None if unknown."""
    u = usage_dict(usage)
    if not u:
        return None
    rates = TOKEN_RATES.get(model, DEFAULT_RATES)
    details = u.get("input_tokens_details") or {}
    image_in = details.get("image_tokens")
    text_in = details.get("text_tokens")
    if image_in is None and text_in is None:
        image_in, text_in = u.get("input_tokens") or 0, 0
    out = u.get("output_tokens") or 0
    return ((image_in or 0) * rates["image_in"] + (text_in or 0) * rates["text_in"]
            + out * rates["image_out"]) / 1_000_000


def _read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def cache_stem(name, model):
    """Cache filename stem: one entry per (prompt row, model)."""
    return f"{name}__{model}"


def _sidecar_path(name, model):
    return CACHE_DIR / f"{cache_stem(name, model)}.json"


def _image_path(name, model):
    return CACHE_DIR / f"{cache_stem(name, model)}.png"


def _load_sidecar(name, model):
    p = _sidecar_path(name, model)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _write_sidecar(name, model, data):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _sidecar_path(name, model).write_text(json.dumps(data, indent=2, default=str))


def do_prompt(row, input_path, force, model):
    name = row["name"]
    prompt_text = row["prompt"]
    quality = EDIT_PARAMS["quality"]
    sidecar = _load_sidecar(name, model)

    if (not force and cache_is_fresh(sidecar, prompt_text, quality)
            and _image_path(name, model).exists()):
        print(f"[lab] '{name}' [{model}]: cached image is fresh; use --force to regenerate.")
        return 0

    from dotenv import load_dotenv

    load_dotenv()
    api_key = resolve_api_key()
    if not api_key:
        print("[lab] No OPENAI_API_KEY found in the environment, a .env file, "
              "or the dissman service unit.")
        return 1
    client = make_client(api_key)

    print(f"[lab] '{name}': calling images.edit on {input_path.name} "
          f"(model={model}, quality={quality}) ...")
    started = time.monotonic()
    try:
        data, usage = edit_image(client, model, input_path, prompt_text)
        elapsed = time.monotonic() - started
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _image_path(name, model).write_bytes(data)
        u = usage_dict(usage)
        cost = estimate_cost(u, model)
        _write_sidecar(name, model, {
            "name": name, "prompt": prompt_text, "input": input_path.name,
            "model": model, "quality": quality, "status": "accepted",
            "usage": u, "est_cost_usd": cost, "seconds": round(elapsed, 1),
            "ts": int(time.time()),
        })
        print(f"[lab] '{name}' [{model}]: ACCEPTED in {elapsed:.1f}s "
              f"-> cached {_image_path(name, model)}")
        cost_txt = f"~${cost:.4f}" if cost is not None else "unknown"
        print(f"      usage={u}  est. cost {cost_txt}")
        return 0
    except Exception as e:  # noqa: BLE001 — we want the raw reason
        status = getattr(e, "status_code", None)
        _write_sidecar(name, model, {
            "name": name, "prompt": prompt_text, "input": input_path.name,
            "model": model, "quality": quality, "status": "rejected",
            "error_type": type(e).__name__, "http_status": status,
            "error": describe_error(e), "ts": int(time.time()),
        })
        print(f"[lab] '{name}' [{model}]: REJECTED ({type(e).__name__}, status={status})")
        print(f"      {describe_error(e)}")
        return 1


def do_print(row, preview, model):
    name = row["name"]
    img_path = _image_path(name, model)
    if not img_path.exists():
        print(f"[lab] '{name}' [{model}]: no cached image. Run --test prompt first.")
        return 1

    from PIL import Image

    settings = PrintSettings.from_row(row)
    label = (f"mode={settings.binarization} t={settings.threshold} "
             f"c={settings.contrast} b={settings.brightness} "
             f"w={settings.resize_width} s={settings.sharpness} g={settings.gamma}")
    processed = render_for_thermal(Image.open(img_path), settings)

    if preview:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        out = PREVIEW_DIR / f"{cache_stem(name, model)}.png"
        processed.save(out)
        print(f"[lab] '{name}': preview written {out}  [{label}]")
        return 0

    from escpos.printer import Usb

    p = Usb(0x0416, 0x5011, in_ep=0x81, out_ep=0x01, profile="POS-5890")
    tmp = CACHE_DIR / f"_print_{cache_stem(name, model)}.png"
    processed.save(tmp)
    try:
        p._raw(b"\x1b\x40")
        p.text(f"\n{name}\n{model}\n{label}\n\n")
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
    ap.add_argument("--model", default=None,
                    help="image model (default: first in prompts/image_models.txt)")
    args = ap.parse_args(argv)
    model = args.model or load_models(MODELS_FILE)[0]

    rows = _read_rows(args.csv)
    row = load_row(rows, args.name)
    if row is None:
        names = ", ".join(r.get("name", "") for r in rows)
        print(f"[lab] no row named '{args.name}'. Available: {names}")
        return 2

    if args.test == "prompt":
        return do_prompt(row, Path(args.input), args.force, model)
    return do_print(row, args.preview, model)


if __name__ == "__main__":
    raise SystemExit(main())
