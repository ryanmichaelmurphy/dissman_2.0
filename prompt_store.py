"""CSV-backed store for GPT-image drawing prompts.

CSV columns: name,weight,is_base,binarization,threshold,contrast,brightness,
resize_width,sharpness,gamma,prompt

The print-variable columns are parsed into a PrintSettings (see print_pipeline)
and returned on the PromptChoice so the chosen prompt's tuned print settings
travel through to the thermal printer.

Selection is weighted-random over the rows whose weight parses to a positive
float. Non-numeric, negative, or zero weights are excluded from the draw.

If all weights are zero/invalid, the row marked is_base=true is returned. If
the CSV is missing or empty, FALLBACK_PROMPT is returned under the name
'fallback'.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

from print_pipeline import PrintSettings


@dataclass(frozen=True)
class PromptChoice:
    name: str
    prompt: str
    settings: PrintSettings


# Keep in lockstep with the is_base row of prompts/drawing_prompts.csv —
# enforced by test_fallback_prompt_matches_repo_csv_base_row. The old harsher
# wording was moderation-blocked in production (softened in commit 8d13e0a).
FALLBACK_PROMPT = (
    "You are a middle schooler joking on your friend. Draw this person as a "
    "crude middle school notebook doodle. Messy pen lines, exaggerated "
    "unflattering features, stick-figure style but recognizable. Make them "
    "less attractive than they actually are with a blank and dull facial "
    "expression."
)


def _parse_weight(raw):
    try:
        w = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return w if w > 0 else 0.0


def _is_truthy(raw):
    return str(raw).strip().lower() in ("true", "1", "yes")


class PromptStore:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def _load(self):
        if not self.csv_path.exists():
            return []
        # utf-8-sig: tolerate a BOM from Excel "CSV UTF-8" saves, which would
        # otherwise turn the first fieldname into '﻿name'.
        with self.csv_path.open(newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def choose(self):
        rows = self._load()
        if not rows:
            reason = "missing" if not self.csv_path.exists() else "no rows"
            # flush=True so the reason reaches the systemd journal immediately.
            print(f"[prompt-store] fallback: {self.csv_path} {reason}", flush=True)
            return PromptChoice("fallback", FALLBACK_PROMPT, PrintSettings.defaults())

        weighted = [(r, _parse_weight(r.get("weight"))) for r in rows]
        positive = [(r, w) for r, w in weighted if w > 0]

        if positive:
            choice = random.choices(
                [r for r, _ in positive],
                weights=[w for _, w in positive],
                k=1,
            )[0]
            return self._to_choice(choice)

        for r in rows:
            if _is_truthy(r.get("is_base")):
                print(f"[prompt-store] no positive weights in {self.csv_path}; "
                      f"using is_base row '{r.get('name', '')}'", flush=True)
                return self._to_choice(r)

        print(f"[prompt-store] fallback: no positive weights and no is_base "
              f"row in {self.csv_path}", flush=True)
        return PromptChoice("fallback", FALLBACK_PROMPT, PrintSettings.defaults())

    def base(self):
        """Return the is_base==true row as a PromptChoice, or None if absent."""
        for r in self._load():
            if _is_truthy(r.get("is_base")):
                return self._to_choice(r)
        return None

    @staticmethod
    def _to_choice(row):
        return PromptChoice(
            row.get("name", ""),
            row.get("prompt", ""),
            PrintSettings.from_row(row),
        )
