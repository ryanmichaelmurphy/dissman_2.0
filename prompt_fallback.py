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
