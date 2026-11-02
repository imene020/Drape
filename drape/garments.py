"""Turn a clothes-segmentation label map into clean, catalog-style crops."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .taxonomy import GARMENT_KINDS, GarmentKind


@dataclass
class Garment:
    kind: GarmentKind
    mask: np.ndarray  # (H, W) bool
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
    area_frac: float
    crop: Image.Image | None = None
    embedding: np.ndarray | None = None
    matches: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.kind.name


def resize_max_side(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return image
    return image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def _largest_blobs(mask: np.ndarray, keep_frac: float = 0.15) -> np.ndarray:
    """Drop speckles: keep connected components >= keep_frac of the biggest.

    Pure numpy flood fill on a downsampled grid keeps this dependency-free and
    fast enough for one image per request.
    """
    h, w = mask.shape
    step = max(1, max(h, w) // 256)
    small = mask[::step, ::step]
    sh, sw = small.shape
    labels = np.zeros((sh, sw), dtype=np.int32)
    sizes: list[int] = [0]
    current = 0
    ys, xs = np.nonzero(small)
    for y0, x0 in zip(ys, xs):
        if labels[y0, x0]:
            continue
        current += 1
        stack = [(y0, x0)]
        labels[y0, x0] = current
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < sh and 0 <= nx < sw and small[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    stack.append((ny, nx))
        sizes.append(size)
    if current <= 1:
        return mask
    biggest = max(sizes)
    keep = {i for i, s in enumerate(sizes) if i and s >= keep_frac * biggest}
    keep_small = np.isin(labels, list(keep))
    keep_full = np.repeat(np.repeat(keep_small, step, axis=0), step, axis=1)[:h, :w]
    return mask & keep_full


def extract_garments(label_map: np.ndarray, min_area_scale: float = 1.0) -> list[Garment]:
    """Group segmentation labels into garments and compute boxes/areas."""
    total = label_map.size
    garments: list[Garment] = []
    for kind in GARMENT_KINDS:
        mask = np.isin(label_map, kind.seg_labels)
        if not mask.any():
            continue
        mask = _largest_blobs(mask) if kind.name not in ("Shoes",) else mask
        area = float(mask.sum()) / total
        if area < kind.min_area * min_area_scale:
            continue
        ys, xs = np.nonzero(mask)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        garments.append(Garment(kind=kind, mask=mask, bbox=bbox, area_frac=area))
    # A dress and a "top + skirt" reading of the same pixels rarely co-occur;
    # if the model splits a dress, keep whichever explanation covers more.
    names = {g.name for g in garments}
    if "Dress" in names and ("Top" in names or "Skirt" in names):
        dress = next(g for g in garments if g.name == "Dress")
        parts = [g for g in garments if g.name in ("Top", "Skirt")]
        if dress.area_frac < 0.5 * sum(p.area_frac for p in parts):
            garments = [g for g in garments if g.name != "Dress"]
    return garments


def make_crop(image: Image.Image, garment: Garment, pad: float = 0.08, min_side: int = 64) -> Image.Image:
    """Cut the garment out onto a white background, like a product shot."""
    image = image.convert("RGB")
    x0, y0, x1, y1 = garment.bbox
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad) + 2, int(h * pad) + 2
    X0, Y0 = max(0, x0 - px), max(0, y0 - py)
    X1, Y1 = min(image.width, x1 + px), min(image.height, y1 + py)

    alpha = Image.fromarray((garment.mask * 255).astype(np.uint8), mode="L")
    radius = max(1, int(round(min(image.size) / 300)))
    alpha = alpha.filter(ImageFilter.MaxFilter(2 * radius + 1)).filter(ImageFilter.GaussianBlur(radius))

    white = Image.new("RGB", image.size, (255, 255, 255))
    cut = Image.composite(image, white, alpha).crop((X0, Y0, X1, Y1))

    # Square canvas so the embedder's centre-crop never chops the garment.
    side = max(cut.width, cut.height, min_side)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(cut, ((side - cut.width) // 2, (side - cut.height) // 2))
    return canvas


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_overlay(image: Image.Image, garments: list[Garment], opacity: float = 0.42) -> Image.Image:
    """Tint each garment and tag it with its name."""
    base = image.convert("RGB")
    tint = np.asarray(base, dtype=np.float32).copy()
    for g in garments:
        colour = np.array(g.kind.color, dtype=np.float32)
        tint[g.mask] = tint[g.mask] * (1 - opacity) + colour * opacity
    out = Image.fromarray(tint.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    font_size = max(12, min(out.size) // 32)
    font = _font(font_size)
    for i, g in enumerate(garments, start=1):
        x0, y0, x1, y1 = g.bbox
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=g.kind.color, width=max(2, font_size // 8))
        label = f"{i} · {g.name}"
        tx, ty = x0 + 4, max(0, y0 - font_size - 8)
        try:
            l, t, r, b = draw.textbbox((tx, ty), label, font=font)
        except AttributeError:  # very old Pillow
            r, b = tx + font_size * len(label) // 2, ty + font_size
            l, t = tx, ty
        draw.rectangle((l - 4, t - 3, r + 4, b + 3), fill=g.kind.color)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return out
