"""Small image utilities shared by the index builder and the benchmark."""

from __future__ import annotations

import io

from PIL import Image


def to_rgb_on_white(image: Image.Image) -> Image.Image:
    """Flatten transparency onto white (many product PNGs are transparent)."""
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    return image.convert("RGB")


def thumbnail_bytes(image: Image.Image, side: int = 256, quality: int = 85) -> bytes:
    """Square, white-padded JPEG thumbnail."""
    im = to_rgb_on_white(image)
    im.thumbnail((side, side), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()
