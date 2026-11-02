"""Model wrappers: fashion embeddings (open_clip) and clothes segmentation.

Both wrappers import torch lazily so the rest of the package (and the test
suite) runs without it. Anything with the same methods can be swapped in,
which is how the tests use small fakes.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

DEFAULT_EMBED_MODEL = "hf-hub:Marqo/marqo-fashionSigLIP"
DEFAULT_SEG_MODEL = "mattmdjaga/segformer_b2_clothes"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class Segmenter(Protocol):
    def segment(self, image: Image.Image) -> np.ndarray:
        """Return an (H, W) uint8 label map aligned with ``image``."""
        ...


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _pick_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing
        return "cpu"


def _set_threads() -> None:
    try:
        import torch

        torch.set_num_threads(max(1, os.cpu_count() or 1))
    except Exception:  # pragma: no cover
        pass


class OpenClipEmbedder:
    """Image/text embeddings from any open_clip model.

    ``spec`` is either an ``hf-hub:`` name (e.g. the default Marqo
    FashionSigLIP) or ``"<architecture>|<pretrained tag>"`` such as
    ``"ViT-B-16-SigLIP|webli"`` for the generic baselines in the benchmark.
    """

    def __init__(self, spec: str = DEFAULT_EMBED_MODEL, device: str | None = None, batch_size: int = 32):
        import open_clip
        import torch

        _set_threads()
        self.name = spec
        self.device = _pick_device(device)
        self.batch_size = batch_size
        if "|" in spec:
            arch, pretrained = spec.split("|", 1)
            model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
            tokenizer = open_clip.get_tokenizer(arch)
        else:
            model, _, preprocess = open_clip.create_model_and_transforms(spec)
            tokenizer = open_clip.get_tokenizer(spec)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self._torch = torch
        with torch.inference_mode():
            probe = self.model.encode_text(self.tokenizer(["probe"]).to(self.device))
        self.dim = int(probe.shape[-1])

    def embed_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = [self.preprocess(im.convert("RGB")) for im in images[start : start + self.batch_size]]
            if not batch:
                continue
            x = torch.stack(batch).to(self.device)
            with torch.inference_mode():
                feats = self.model.encode_image(x)
            out.append(feats.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.concatenate(out))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(list(texts[start : start + self.batch_size])).to(self.device)
            with torch.inference_mode():
                feats = self.model.encode_text(tokens)
            out.append(feats.float().cpu().numpy())
        if not out:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.concatenate(out))


class SegformerClothesSegmenter:
    """Pixel-level clothes parsing with a SegFormer fine-tuned on ATR."""

    def __init__(self, model_id: str = DEFAULT_SEG_MODEL, device: str | None = None, model=None, processor=None):
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

        _set_threads()
        self.device = _pick_device(device)
        self.processor = processor or SegformerImageProcessor.from_pretrained(model_id)
        self.model = (model or AutoModelForSemanticSegmentation.from_pretrained(model_id)).to(self.device).eval()
        self._torch = torch

    def segment(self, image: Image.Image) -> np.ndarray:
        torch = self._torch
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits  # (1, C, h/4, w/4)
        upsampled = torch.nn.functional.interpolate(
            logits, size=(image.height, image.width), mode="bilinear", align_corners=False
        )
        return upsampled.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
