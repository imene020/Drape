"""Photo-condition augmentations for the retrieval benchmark.

A query is a catalog product shot degraded the way a garment looks when
cropped out of a real photo: off-centre framing, tilt, lighting and colour
cast, blur, compression and, in the "wild" setting, a non-white background
and a partial occlusion (an arm, hair, a bag strap).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _random_resized_crop(im: Image.Image, rng: np.random.Generator) -> Image.Image:
    w, h = im.size
    area = rng.uniform(0.72, 0.95) * w * h
    ratio = np.exp(rng.uniform(np.log(0.8), np.log(1.25)))
    cw = int(min(w, round(np.sqrt(area * ratio))))
    ch = int(min(h, round(np.sqrt(area / ratio))))
    x = int(rng.integers(0, w - cw + 1))
    y = int(rng.integers(0, h - ch + 1))
    return im.crop((x, y, x + cw, y + ch))


def _background(size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    w, h = size
    c1 = rng.uniform(30, 225, size=3)
    c2 = rng.uniform(30, 225, size=3)
    t = np.linspace(0, 1, h)[:, None, None] if rng.random() < 0.5 else np.linspace(0, 1, w)[None, :, None]
    grad = c1 * (1 - t) + c2 * t
    grad = np.broadcast_to(grad, (h, w, 3)).copy()
    grad += rng.normal(0, rng.uniform(2, 14), size=(h, w, 3))
    return grad.clip(0, 255)


def augment(im: Image.Image, rng: np.random.Generator, wild: bool = False) -> Image.Image:
    im = im.convert("RGB")
    if wild:
        arr = np.asarray(im, dtype=np.float32)
        white = (arr > 236).all(axis=2)
        bg = _background(im.size, rng)
        arr[white] = bg[white]
        im = Image.fromarray(arr.astype(np.uint8))

    im = _random_resized_crop(im, rng)
    angle = float(rng.uniform(-12, 12))
    fill = tuple(int(v) for v in np.asarray(im).reshape(-1, 3)[0]) if wild else (255, 255, 255)
    im = im.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)

    im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.75, 1.25)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.75, 1.25)))
    im = ImageEnhance.Color(im).enhance(float(rng.uniform(0.7, 1.3)))
    # Warm/cool colour cast (indoor vs daylight white balance).
    cast = np.array([1 + rng.uniform(-0.08, 0.08), 1.0, 1 + rng.uniform(-0.08, 0.08)], dtype=np.float32)
    im = Image.fromarray((np.asarray(im, dtype=np.float32) * cast).clip(0, 255).astype(np.uint8))

    if wild:
        w, h = im.size
        ow = int(w * rng.uniform(0.25, 0.45))
        oh = int(h * rng.uniform(0.25, 0.45))
        x, y = int(rng.integers(0, w - ow + 1)), int(rng.integers(0, h - oh + 1))
        colour = tuple(int(c) for c in rng.uniform(20, 235, size=3))
        occ = Image.new("RGB", (ow, oh), colour)
        im.paste(occ, (x, y))

    blur = float(rng.uniform(0, 1.2))
    if blur > 0.2:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    side = int(rng.integers(160, 257))
    im.thumbnail((side, side), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(rng.integers(35, 76)))
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
