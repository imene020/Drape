"""Tiny deterministic stand-ins for the real models."""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image

from drape.models import l2_normalize

DIM = 48


class FakeEmbedder:
    """Colour-histogram 'embedding'. Same colours -> close vectors."""

    name = "fake"
    dim = DIM

    def embed_images(self, images):
        out = []
        for im in images:
            arr = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32) / 255.0
            pix = arr.reshape(-1, 3)
            fg = pix[(pix < 0.94).any(axis=1)]
            if len(fg) == 0:
                fg = pix
            hist, _ = np.histogramdd(fg, bins=(3, 3, 3), range=((0, 1), (0, 1), (0, 1)))
            vec = np.concatenate([hist.ravel() / len(fg), fg.mean(axis=0), np.zeros(DIM - 30)])
            out.append(vec)
        return l2_normalize(np.stack(out)) if out else np.zeros((0, DIM), np.float32)

    def embed_texts(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            out.append(np.random.default_rng(seed).normal(size=DIM))
        return l2_normalize(np.stack(out)) if out else np.zeros((0, DIM), np.float32)


class FakeSegmenter:
    """Returns a fixed label map, resized to the image."""

    def __init__(self, label_map: np.ndarray):
        self.label_map = label_map

    def segment(self, image):
        lm = Image.fromarray(self.label_map.astype(np.uint8))
        return np.asarray(lm.resize(image.size, Image.NEAREST))


def solid(colour, size=(64, 64)) -> Image.Image:
    return Image.new("RGB", size, colour)


def product(colour, size=(64, 64)) -> Image.Image:
    """A coloured square on white, like a product shot."""
    im = Image.new("RGB", size, (255, 255, 255))
    w, h = size
    im.paste(Image.new("RGB", (w // 2, h // 2), colour), (w // 4, h // 4))
    return im
