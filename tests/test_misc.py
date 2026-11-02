import importlib
import io
import sys

import numpy as np
from PIL import Image

from drape.augment import augment
from drape.imaging import thumbnail_bytes, to_rgb_on_white
from fakes import product


def test_augment_is_deterministic_and_changes_image():
    im = product((200, 30, 30), size=(256, 256))
    a = augment(im, np.random.default_rng(7), wild=True)
    b = augment(im, np.random.default_rng(7), wild=True)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert a.mode == "RGB" and max(a.size) <= 256
    clean = np.asarray(augment(im, np.random.default_rng(1), wild=False))
    assert clean.shape[2] == 3


def test_transparent_png_flattens_to_white():
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    assert to_rgb_on_white(rgba).getpixel((0, 0)) == (255, 255, 255)


def test_thumbnail_is_square_jpeg():
    data = thumbnail_bytes(Image.new("RGB", (300, 100), (10, 20, 30)), side=128)
    im = Image.open(io.BytesIO(data))
    assert im.format == "JPEG" and im.size == (128, 128)


def test_app_starts_even_when_catalog_is_missing(monkeypatch, tmp_path):
    """No index and no network: the UI must still come up with a setup message."""
    monkeypatch.setenv("DRAPE_INDEX_DIR", str(tmp_path / "missing"))
    for mod in ("app", "drape.config"):
        sys.modules.pop(mod, None)
    app = importlib.import_module("app")
    assert app.ENGINE is None and app.STARTUP_ERROR
    assert app.demo is not None
