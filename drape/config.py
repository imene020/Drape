"""Runtime settings, all overridable with environment variables (Space variables)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .models import DEFAULT_EMBED_MODEL, DEFAULT_SEG_MODEL


@dataclass(frozen=True)
class Settings:
    index_repo: str = os.getenv("DRAPE_INDEX_REPO", "Imene-ee/drape-catalog")
    index_dir: str | None = os.getenv("DRAPE_INDEX_DIR") or None  # local index wins over the Hub
    embed_model: str = os.getenv("DRAPE_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    seg_model: str = os.getenv("DRAPE_SEG_MODEL", DEFAULT_SEG_MODEL)
    max_side: int = int(os.getenv("DRAPE_MAX_SIDE", "1024"))
    k: int = int(os.getenv("DRAPE_TOP_K", "8"))


settings = Settings()
