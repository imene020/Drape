"""The product catalog: embeddings, metadata, thumbnails and search."""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from .models import l2_normalize

EMB_FILE = "embeddings.npy"
META_FILE = "catalog.parquet"
MANIFEST_FILE = "manifest.json"


@dataclass
class Match:
    row: int
    item_id: str
    title: str
    group: str
    score: float  # cosine similarity (or fused score) used for ranking
    similarity: float | None  # cosine to the visual query, for display (None for text search)

    def caption(self) -> str:
        title = (self.title or "").strip()
        if len(title) > 48:
            title = title[:47].rstrip() + "…"
        if self.similarity is None:
            return title or self.item_id
        pct = int(round(max(0.0, min(1.0, self.similarity)) * 100))
        return f"{title} · {pct}% similar" if title else f"{pct}% similar"


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    sd = float(x.std())
    return (x - float(x.mean())) / (sd if sd > 1e-6 else 1.0)


def mmr(
    candidates: np.ndarray,
    scores: np.ndarray,
    embeddings: np.ndarray,
    k: int,
    lam: float = 0.8,
    dup_threshold: float = 0.97,
) -> list[int]:
    """Maximal marginal relevance: relevant but not near-duplicates.

    ``candidates`` are row ids, ``scores`` their relevance (same order),
    ``embeddings`` the full normalized matrix. Relevance is min-max scaled to
    [0, 1] so it is comparable with cosine redundancy whatever the score
    scale; candidates nearly identical to an already chosen item
    (cosine >= ``dup_threshold``, e.g. the same product listed twice) are
    skipped while anything else is left.
    """
    if len(candidates) == 0:
        return []
    k = min(k, len(candidates))
    cand_emb = embeddings[candidates].astype(np.float32)
    s = np.asarray(scores, dtype=np.float32)
    span = float(s.max() - s.min())
    rel = (s - s.min()) / span if span > 1e-9 else np.ones_like(s)
    chosen: list[int] = [int(np.argmax(rel))]
    max_sim = cand_emb @ cand_emb[chosen[0]]
    while len(chosen) < k:
        value = lam * rel - (1 - lam) * max_sim
        value[chosen] = -np.inf
        dup = max_sim >= dup_threshold
        if (~dup & np.isfinite(value)).any():
            value[dup] = -np.inf
        nxt = int(np.argmax(value))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, cand_emb @ cand_emb[nxt])
    return [int(candidates[i]) for i in chosen]


class CatalogIndex:
    """In-memory vector index. Brute-force cosine is exact and, at catalog
    sizes up to a few hundred thousand items, fast enough on one CPU core."""

    def __init__(self, embeddings: np.ndarray, meta: pd.DataFrame, manifest: dict | None = None):
        if len(embeddings) != len(meta):
            raise ValueError(f"{len(embeddings)} embeddings but {len(meta)} catalog rows")
        for col in ("item_id", "title", "group"):
            if col not in meta.columns:
                raise ValueError(f"catalog is missing column '{col}'")
        self.embeddings = l2_normalize(embeddings).astype(np.float32)
        self.meta = meta.reset_index(drop=True)
        self.manifest = manifest or {}
        self.groups = self.meta["group"].astype(str).to_numpy()
        self._group_rows = {g: np.flatnonzero(self.groups == g) for g in np.unique(self.groups)}

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, directory: str | os.PathLike) -> "CatalogIndex":
        directory = Path(directory)
        emb = np.load(directory / EMB_FILE)
        meta = pd.read_parquet(directory / META_FILE)
        manifest_path = directory / MANIFEST_FILE
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        return cls(emb, meta, manifest)

    @classmethod
    def from_hub(cls, repo_id: str, revision: str | None = None) -> "CatalogIndex":
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[EMB_FILE, META_FILE, MANIFEST_FILE],
        )
        return cls.load(path)

    def save(self, directory: str | os.PathLike) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / EMB_FILE, self.embeddings.astype(np.float16))
        self.meta.to_parquet(directory / META_FILE, index=False)
        (directory / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2))

    # ------------------------------------------------------------------ props
    def __len__(self) -> int:
        return len(self.meta)

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def group_counts(self) -> dict[str, int]:
        return {g: int(len(r)) for g, r in sorted(self._group_rows.items())}

    def rows_for(self, groups: Iterable[str] | None) -> np.ndarray:
        if not groups:
            return np.arange(len(self))
        parts = [self._group_rows[g] for g in groups if g in self._group_rows]
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.sort(np.concatenate(parts))

    def thumbnail(self, row: int) -> Image.Image:
        data = self.meta.at[row, "thumb"] if "thumb" in self.meta.columns else None
        if data is None or (isinstance(data, float) and np.isnan(data)):
            return Image.new("RGB", (256, 256), (240, 238, 234))
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _match(self, row: int, score: float, similarity: float | None) -> Match:
        r = self.meta.iloc[row]
        return Match(
            row=int(row),
            item_id=str(r["item_id"]),
            title=str(r["title"] or ""),
            group=str(r["group"]),
            score=float(score),
            similarity=None if similarity is None else float(similarity),
        )

    # ----------------------------------------------------------------- search
    def search(
        self,
        query: np.ndarray,
        k: int = 8,
        groups: Sequence[str] | None = None,
        text_query: np.ndarray | None = None,
        alpha: float = 0.0,
        exclude: Iterable[int] = (),
        diversify: bool = True,
        pool: int = 120,
    ) -> list[Match]:
        """Nearest catalog items to ``query``.

        If ``text_query`` is given, image and text similarities are z-scored
        over the candidate pool and blended with weight ``alpha`` on the
        text side ("this jacket, but in black leather"). Z-scoring matters:
        image-image cosines sit around 0.6-0.9 while image-text cosines sit
        around 0.05-0.2, so adding raw scores would let one side drown the
        other.
        """
        rows = self.rows_for(groups)
        if len(rows) == 0:
            return []
        excl = set(int(e) for e in exclude)
        if excl:
            rows = np.array([r for r in rows if r not in excl], dtype=np.int64)
            if len(rows) == 0:
                return []
        emb = self.embeddings[rows]
        q = l2_normalize(np.asarray(query, dtype=np.float32).reshape(-1))
        sim = emb @ q
        if text_query is not None and alpha > 0:
            t = l2_normalize(np.asarray(text_query, dtype=np.float32).reshape(-1))
            tsim = emb @ t
            score = (1 - alpha) * zscore(sim) + alpha * zscore(tsim)
        else:
            score = sim
        top = np.argsort(-score)[: max(k, pool if diversify else k)]
        if diversify and len(top) > k:
            picked_rows = mmr(rows[top], score[top], self.embeddings, k)
            pos = {int(r): i for i, r in enumerate(rows)}
            return [self._match(r, score[pos[r]], sim[pos[r]]) for r in picked_rows]
        return [self._match(rows[i], score[i], sim[i]) for i in top[:k]]

    def search_text(self, text_query: np.ndarray, k: int = 12, groups: Sequence[str] | None = None) -> list[Match]:
        rows = self.rows_for(groups)
        if len(rows) == 0:
            return []
        t = l2_normalize(np.asarray(text_query, dtype=np.float32).reshape(-1))
        sim = self.embeddings[rows] @ t
        top = np.argsort(-sim)[: max(k, 120)]
        picked = mmr(rows[top], sim[top], self.embeddings, k, lam=0.85)
        pos = {int(r): i for i, r in enumerate(rows)}
        # Text-image cosines are on a different scale from image-image ones,
        # so no percentage is shown for text search.
        return [self._match(r, sim[pos[r]], None) for r in picked]
