"""Image-model roster and the single image-edit call for production + the lab.

OpenAI retires image models regularly (gpt-image-1 was shut down 2026-10-23),
so the model is not hardcoded: prompts/image_models.txt lists models in
preference order and dissman.py falls through them when one is retired or
unavailable. Both dissman.py and prompt_testing/lab.py call edit_image(), so
what the lab tests is exactly what the kiosk sends.

Pure except for make_client() (which imports the SDK); everything else is
unit-tested in tests/test_image_models.py.
"""

from __future__ import annotations

import base64
from pathlib import Path

import openai

# Fallback when prompts/image_models.txt is missing/empty. Keep in sync with it.
DEFAULT_MODELS = ["gpt-image-2.5-flare", "gpt-image-2.5-sunburst", "gpt-image-1-mini"]

# Thermal output is resized to ~380px and binarized, so "low" quality is
# indistinguishable on paper and much cheaper/faster than the default "auto".
EDIT_PARAMS = {"n": 1, "size": "1024x1024", "quality": "low"}

# Per-attempt HTTP timeout (s). The SDK default is 600s x 2 retries, which could
# leave LoadScreen spinning for ~30 minutes on a hung connection.
REQUEST_TIMEOUT = 90
MAX_RETRIES = 1


def load_models(path) -> list[str]:
    """Models from `path` in order (one per line, # comments). Never raises."""
    try:
        lines = Path(path).read_text().splitlines()
    except (OSError, UnicodeDecodeError):
        return list(DEFAULT_MODELS)
    models = []
    for line in lines:
        name = line.split("#", 1)[0].strip()
        if name and name not in models:
            models.append(name)
    return models or list(DEFAULT_MODELS)


def make_client(api_key):
    return openai.OpenAI(api_key=api_key, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)


def edit_image(client, model, image_path, prompt):
    """One images.edit call. Returns (png_bytes, usage); raises on any failure."""
    with open(image_path, "rb") as f:
        response = client.images.edit(model=model, image=f, prompt=prompt, **EDIT_PARAMS)
    data = getattr(response, "data", None) or []
    b64 = getattr(data[0], "b64_json", None) if data else None
    if not b64:
        raise ValueError(f"no image data in response from {model}")
    return base64.b64decode(b64), getattr(response, "usage", None)


def _error_code(e):
    code = getattr(e, "code", None)
    body = getattr(e, "body", None)
    if not code and isinstance(body, dict):
        code = body.get("code")
        if not code and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
    return code


def classify_error(e) -> str:
    """How the fallback loop should react to a failed edit call.

    "content": this prompt was rejected by moderation; try the next prompt.
    "model":   this model is retired/unavailable/incompatible; try the next model.
    "fatal":   account, network or server problem; no other model/prompt helps.
    """
    if isinstance(e, openai.BadRequestError):
        if _error_code(e) == "moderation_blocked" or "safety" in str(getattr(e, "body", "")).lower():
            return "content"
        return "model"
    if isinstance(e, (openai.NotFoundError, openai.PermissionDeniedError)):
        return "model"
    return "fatal"


def describe_error(e) -> str:
    """One log line with everything needed to diagnose a failure from pi-logs."""
    parts = [f"{type(e).__name__}:"]
    status = getattr(e, "status_code", None)
    if status is not None:
        parts.append(f"status={status}")
    request_id = getattr(e, "request_id", None)
    if request_id:
        parts.append(f"request_id={request_id}")
    code = _error_code(e)
    if code:
        parts.append(f"code={code}")
    body = getattr(e, "body", None)
    parts.append(str(body) if body else str(e))
    if e.__cause__ is not None:
        parts.append(f"cause={e.__cause__!r}")
    return " ".join(parts)


class ModelRoster:
    """Which models are still worth trying this boot (in memory only).

    A model that fails with a "model" error is dropped until the next reboot;
    the last model that succeeded is tried first.
    """

    def __init__(self, models):
        self._models = list(dict.fromkeys(models))
        self._dead = set()
        self._good = None

    def alive(self) -> list[str]:
        live = [m for m in self._models if m not in self._dead]
        if self._good in live:
            live.remove(self._good)
            live.insert(0, self._good)
        return live

    def mark_dead(self, model):
        if model in self._models:
            self._dead.add(model)
        if self._good == model:
            self._good = None

    def mark_good(self, model):
        if model in self._models:
            self._good = model
