"""Shop-the-look pipeline: photo -> garments -> catalog matches -> outfit completion."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .garments import Garment, extract_garments, make_crop, render_overlay, resize_max_side
from .index import CatalogIndex, Match
from .models import Embedder, Segmenter, l2_normalize
from .taxonomy import GARMENT_KINDS, GROUP_PROMPTS, GROUPS, OTHER, GarmentKind, complete_look_targets

# Used when the photo has no person in it (a flat-lay or a product shot):
# the whole image becomes one "garment" of the zero-shot-predicted group.
_ITEM_KIND_BY_GROUP: dict[str, GarmentKind] = {}
for _k in GARMENT_KINDS:
    _ITEM_KIND_BY_GROUP.setdefault(_k.search_groups[0], _k)
_ITEM_KIND_BY_GROUP["outerwear"] = GarmentKind("Outerwear", (), ("outerwear", "top"), 0.0, (160, 110, 80))
_ITEM_KIND_BY_GROUP["accessory"] = GarmentKind("Accessory", (), ("accessory",), 0.0, (200, 100, 130))
_ITEM_KIND_BY_GROUP["bottom"] = GarmentKind("Bottom", (), ("bottom",), 0.0, (90, 140, 200))


@dataclass
class LookResult:
    image: Image.Image  # resized input
    overlay: Image.Image
    garments: list[Garment]
    look_embedding: np.ndarray
    mode: str  # "outfit" (person found) or "item" (single product)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def kind_names(self) -> list[str]:
        return [g.name for g in self.garments]

    @property
    def present_groups(self) -> list[str]:
        return [g.kind.search_groups[0] for g in self.garments]


class ShopTheLook:
    def __init__(self, embedder: Embedder, segmenter: Segmenter, index: CatalogIndex, max_side: int = 1024):
        self.embedder = embedder
        self.segmenter = segmenter
        self.index = index
        self.max_side = max_side
        self._group_names: list[str] = []
        self._group_protos: np.ndarray | None = None

    # ------------------------------------------------------------ zero-shot
    def _group_prototypes(self) -> tuple[list[str], np.ndarray]:
        if self._group_protos is None:
            names, protos = [], []
            for group in (*GROUPS, OTHER):
                emb = self.embedder.embed_texts(GROUP_PROMPTS[group])
                names.append(group)
                protos.append(l2_normalize(emb.mean(axis=0)))
            self._group_names, self._group_protos = names, np.stack(protos)
        return self._group_names, self._group_protos

    def classify_group(self, embedding: np.ndarray) -> str:
        names, protos = self._group_prototypes()
        scores = protos @ l2_normalize(embedding.reshape(-1))
        order = np.argsort(-scores)
        for i in order:
            if names[i] != OTHER:
                return names[i]
        return names[order[0]]

    # --------------------------------------------------------------- analyze
    def analyze(self, image: Image.Image, k: int = 8, min_area_scale: float = 1.0) -> LookResult:
        t0 = time.perf_counter()
        image = resize_max_side(image, self.max_side)
        label_map = self.segmenter.segment(image)
        t1 = time.perf_counter()
        garments = extract_garments(label_map, min_area_scale=min_area_scale)
        # A person shot with no garment we can match (e.g. only hair/skin) is
        # handled like a product shot.
        mode = "outfit" if garments else "item"

        crops = [make_crop(image, g) for g in garments]
        embs = self.embedder.embed_images([*crops, image])
        t2 = time.perf_counter()
        full_emb = embs[-1]

        if mode == "item":
            group = self.classify_group(full_emb)
            kind = _ITEM_KIND_BY_GROUP.get(group, _ITEM_KIND_BY_GROUP["top"])
            h, w = image.height, image.width
            g = Garment(kind=kind, mask=np.ones((h, w), dtype=bool), bbox=(0, 0, w, h), area_frac=1.0)
            g.crop, g.embedding = image, full_emb
            garments = [g]
            look = full_emb
        else:
            for g, crop, emb in zip(garments, crops, embs[:-1]):
                g.crop, g.embedding = crop, emb
            look = l2_normalize(0.5 * full_emb + 0.5 * l2_normalize(np.mean(embs[:-1], axis=0)))

        for g in garments:
            g.matches = self.index.search(g.embedding, k=k, groups=g.kind.search_groups)
        t3 = time.perf_counter()

        overlay = render_overlay(image, garments) if mode == "outfit" else image
        return LookResult(
            image=image,
            overlay=overlay,
            garments=garments,
            look_embedding=look,
            mode=mode,
            timings={"segment_s": t1 - t0, "embed_s": t2 - t1, "search_s": t3 - t2, "total_s": t3 - t0},
        )

    # ---------------------------------------------------------------- refine
    def refine(
        self,
        embedding: np.ndarray,
        groups: tuple[str, ...],
        text: str,
        alpha: float = 0.5,
        k: int = 8,
    ) -> list[Match]:
        text = (text or "").strip()
        if not text:
            return self.index.search(embedding, k=k, groups=groups)
        t = self.embedder.embed_texts([text])[0]
        return self.index.search(embedding, k=k, groups=groups, text_query=t, alpha=alpha)

    # -------------------------------------------------------- complete look
    def complete_look(
        self,
        look_embedding: np.ndarray,
        present_groups: list[str],
        k_per_group: int = 4,
        style_text: str = "",
        style_alpha: float = 0.35,
        exclude: tuple[int, ...] = (),
    ) -> dict[str, list[Match]]:
        """Style-coherent suggestions for the pieces the outfit is missing.

        Zero-shot: fashion embeddings place items with shared colour, material
        and formality close together even across categories, so the nearest
        items to the *whole look* inside a missing category tend to go with it.
        """
        style_text = (style_text or "").strip()
        t = self.embedder.embed_texts([style_text])[0] if style_text else None
        out: dict[str, list[Match]] = {}
        for group in complete_look_targets(present_groups):
            out[group] = self.index.search(
                look_embedding,
                k=k_per_group,
                groups=(group,),
                text_query=t,
                alpha=style_alpha if t is not None else 0.0,
                exclude=exclude,
            )
        return {g: m for g, m in out.items() if m}

    # ----------------------------------------------------------- text search
    def text_search(self, text: str, k: int = 12, group: str | None = None) -> list[Match]:
        text = (text or "").strip()
        if not text:
            return []
        t = self.embedder.embed_texts([text])[0]
        return self.index.search_text(t, k=k, groups=(group,) if group else None)
