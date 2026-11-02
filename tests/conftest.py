import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from drape.imaging import thumbnail_bytes  # noqa: E402
from drape.index import CatalogIndex  # noqa: E402
from fakes import FakeEmbedder, product  # noqa: E402

# A tiny catalog: 3 colours per group, so colour-matching inside a group is testable.
COLOURS = {
    "red": (200, 30, 30),
    "blue": (30, 60, 200),
    "green": (30, 160, 60),
}
GROUP_ROWS = ["top", "outerwear", "dress", "bottom", "shoes", "bag", "hat", "accessory"]


@pytest.fixture(scope="session")
def embedder():
    return FakeEmbedder()


@pytest.fixture(scope="session")
def catalog(embedder) -> CatalogIndex:
    rows, images = [], []
    for g in GROUP_ROWS:
        for cname, rgb in COLOURS.items():
            im = product(rgb)
            images.append(im)
            rows.append({"item_id": f"{g}-{cname}", "title": f"{cname} {g}", "category": g, "group": g,
                         "group_kw": g, "group_zs": g, "thumb": thumbnail_bytes(im, side=64)})
    emb = embedder.embed_images(images)
    return CatalogIndex(emb, pd.DataFrame(rows), {"embed_model": "fake"})


@pytest.fixture
def outfit_label_map() -> np.ndarray:
    """100x60 'person': top (4), pants (6), shoes (9, 10), bag (16)."""
    lm = np.zeros((100, 60), dtype=np.uint8)
    lm[10:45, 15:45] = 4
    lm[45:85, 18:42] = 6
    lm[85:95, 18:28] = 9
    lm[85:95, 32:42] = 10
    lm[40:60, 46:58] = 16
    lm[2:10, 22:38] = 2  # hair: not a garment
    return lm


@pytest.fixture
def outfit_image(outfit_label_map) -> "Image.Image":
    from PIL import Image

    colours = {0: (245, 245, 245), 2: (40, 30, 20), 4: (200, 30, 30), 6: (30, 60, 200),
               9: (30, 160, 60), 10: (30, 160, 60), 16: (200, 30, 30)}
    arr = np.zeros((*outfit_label_map.shape, 3), dtype=np.uint8)
    for label, rgb in colours.items():
        arr[outfit_label_map == label] = rgb
    return Image.fromarray(arr)
