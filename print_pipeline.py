"""Thermal print rendering, shared by the lab CLI and dissman.py.

PrintSettings mirrors the per-prompt columns in prompts/drawing_prompts.csv.
render_for_thermal turns an RGB/L image into the 1-bit (mode "1") image the
POS-5890 prints, so what the lab tunes is exactly what production prints.

PIL is imported lazily inside render_for_thermal so importing PrintSettings
(e.g. from prompt_store) does not require Pillow.
"""

from __future__ import annotations

from dataclasses import dataclass

_BINARIZATION_MODES = ("dither", "threshold")


@dataclass(frozen=True)
class PrintSettings:
    binarization: str = "dither"
    threshold: int = 128
    contrast: float = 0.3
    brightness: float = 1.0
    resize_width: int = 380
    sharpness: float = 1.0
    gamma: float = 1.0

    @classmethod
    def defaults(cls) -> "PrintSettings":
        return cls()

    @classmethod
    def from_row(cls, row: dict) -> "PrintSettings":
        def _f(key, default, lo, hi):
            try:
                v = float(row.get(key, ""))
            except (TypeError, ValueError):
                return default
            return max(lo, min(hi, v))

        def _i(key, default, lo, hi):
            return int(round(_f(key, default, lo, hi)))

        mode = str(row.get("binarization", "")).strip().lower()
        if mode not in _BINARIZATION_MODES:
            mode = "dither"

        return cls(
            binarization=mode,
            threshold=_i("threshold", 128, 0, 255),
            contrast=_f("contrast", 0.3, 0.0, 3.0),
            brightness=_f("brightness", 1.0, 0.0, 2.0),
            resize_width=_i("resize_width", 380, 1, 384),
            sharpness=_f("sharpness", 1.0, 0.0, 3.0),
            gamma=_f("gamma", 1.0, 0.1, 3.0),
        )


def render_for_thermal(image, settings: PrintSettings):
    """Return a mode "1" image ready for escpos p.image()."""
    from PIL import Image, ImageEnhance

    img = image.convert("L")

    w = max(1, int(settings.resize_width))
    h = max(1, round(img.height * w / img.width))
    img = img.resize((w, h), Image.Resampling.LANCZOS)

    if settings.contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(settings.contrast)
    if settings.brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(settings.brightness)
    if settings.sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(settings.sharpness)
    if settings.gamma != 1.0:
        g = settings.gamma
        lut = [int(round(255 * (i / 255) ** g)) for i in range(256)]
        img = img.point(lut)

    if settings.binarization == "threshold":
        t = settings.threshold
        img = img.point(lambda p, t=t: 255 if p >= t else 0)
        img = img.convert("1", dither=Image.Dither.NONE)
    else:
        img = img.convert("1")  # Floyd–Steinberg (PIL default)

    return img
