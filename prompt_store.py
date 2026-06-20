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


FALLBACK_PROMPT = (
    "You are a middle school bully. Draw this person as a crude middle school "
    "notebook doodle. Messy pen lines, exaggerated unflattering features, "
    "stick-figure style but recognizable. Make them uglier than they actually "
    "are with a stupid facial expression."
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
        with self.csv_path.open(newline="") as f:
            return list(csv.DictReader(f))

    def choose(self):
        rows = self._load()
        if not rows:
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
                return self._to_choice(r)

        return PromptChoice("fallback", FALLBACK_PROMPT, PrintSettings.defaults())

    @staticmethod
    def _to_choice(row):
        return PromptChoice(
            row.get("name", ""),
            row.get("prompt", ""),
            PrintSettings.from_row(row),
        )
