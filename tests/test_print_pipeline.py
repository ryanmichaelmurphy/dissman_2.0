from PIL import Image

from print_pipeline import PrintSettings, render_for_thermal


def _ramp(width=256, height=10):
    """L-mode image where column x has gray value x."""
    img = Image.new("L", (width, height))
    img.putdata([x for _ in range(height) for x in range(width)])
    return img


def test_defaults_match_todays_behavior():
    s = PrintSettings.defaults()
    assert s.binarization == "dither"
    assert s.threshold == 128
    assert s.contrast == 0.3
    assert s.brightness == 1.0
    assert s.resize_width == 380
    assert s.sharpness == 1.0
    assert s.gamma == 1.0


def test_from_row_parses_and_clamps():
    s = PrintSettings.from_row({
        "binarization": "THRESHOLD", "threshold": "300", "contrast": "1.5",
        "brightness": "0.9", "resize_width": "384", "sharpness": "2.0",
        "gamma": "1.2",
    })
    assert s.binarization == "threshold"
    assert s.threshold == 255          # clamped 0-255
    assert s.contrast == 1.5
    assert s.resize_width == 384


def test_from_row_uses_defaults_for_missing_or_bad():
    s = PrintSettings.from_row({"binarization": "nonsense", "contrast": "oops"})
    assert s.binarization == "dither"  # unknown mode -> default
    assert s.contrast == 0.3           # unparseable -> default
    assert s.threshold == 128


def test_render_returns_one_bit_image():
    out = render_for_thermal(_ramp(), PrintSettings.defaults())
    assert out.mode == "1"


def test_threshold_binarization_splits_at_cutoff():
    s = PrintSettings(binarization="threshold", threshold=128, contrast=1.0,
                      brightness=1.0, resize_width=256, sharpness=1.0, gamma=1.0)
    out = render_for_thermal(_ramp(256, 10), s)
    assert out.getpixel((50, 5)) == 0      # below cutoff -> black
    assert out.getpixel((200, 5)) == 255   # above cutoff -> white


def test_dither_produces_only_black_and_white():
    gray = Image.new("L", (64, 64), 128)
    s = PrintSettings(binarization="dither", threshold=128, contrast=1.0,
                      brightness=1.0, resize_width=64, sharpness=1.0, gamma=1.0)
    out = render_for_thermal(gray, s)
    values = set(out.getdata())
    assert values <= {0, 255}
    assert 0 in values and 255 in values   # mid-gray actually dithers


def test_resize_width_is_honored():
    s = PrintSettings.from_row({"resize_width": "200"})
    out = render_for_thermal(_ramp(512, 512), s)
    assert out.width == 200
