"""Garment taxonomy shared by the segmenter, the catalog and the UI.

Two vocabularies meet here:

* **Garment kinds** come from the clothes-segmentation model (what we can see
  on a person: "Top", "Pants", "Shoes", ...).
* **Catalog groups** are the coarse shelves the product catalog is sorted into
  ("top", "outerwear", "bottom", ...).

Every detected garment kind maps to one or more catalog groups it is allowed
to match, which is what keeps a shoe crop from ever returning a handbag.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------
# Catalog groups
# --------------------------------------------------------------------------

GROUPS: tuple[str, ...] = (
    "top",
    "outerwear",
    "dress",
    "bottom",
    "shoes",
    "bag",
    "hat",
    "accessory",
)
OTHER = "other"  # beauty, home, etc. Kept out of the searchable catalog.

GROUP_LABELS: dict[str, str] = {
    "top": "Tops",
    "outerwear": "Outerwear",
    "dress": "Dresses",
    "bottom": "Bottoms",
    "shoes": "Shoes",
    "bag": "Bags",
    "hat": "Hats",
    "accessory": "Accessories",
}

# Zero-shot prompts used when a catalog item has no usable category text, and
# when a query photo contains no person (single product shot).
GROUP_PROMPTS: dict[str, list[str]] = {
    "top": ["a photo of a top", "a t-shirt", "a blouse", "a sweater", "a shirt"],
    "outerwear": ["a jacket", "a coat", "a blazer", "a trench coat"],
    "dress": ["a dress", "a jumpsuit", "an evening gown"],
    "bottom": ["a pair of trousers", "a pair of jeans", "a skirt", "a pair of shorts"],
    "shoes": ["a pair of shoes", "sneakers", "boots", "high heels", "sandals"],
    "bag": ["a handbag", "a purse", "a backpack", "a tote bag"],
    "hat": ["a hat", "a baseball cap", "a beanie"],
    "accessory": ["a necklace", "earrings", "a scarf", "a belt", "sunglasses", "a watch"],
    OTHER: ["a lipstick", "a bottle of perfume", "a makeup palette", "a candle", "a home decor item"],
}

# Keyword rules for catalog category / title text. Order matters: the first
# group whose pattern matches wins ("shirt dress" is a dress, "denim jacket"
# is outerwear, "top-handle bag" is a bag).
_KEYWORDS: list[tuple[str, list[str]]] = [
    (OTHER, ["makeup", "lipstick", "lip gloss", "nail polish", "nail", "fragrance", "perfume",
             "eau de", "beauty", "skincare", "skin care", "cosmetic", "eyeshadow", "mascara",
             "foundation", "blush", "bronzer", "candle", "decor", "home", "pillow", "mug",
             "phone case", "iphone", "book"]),
    ("dress", ["dress", "dresses", "gown", "jumpsuit", "romper", "playsuit", "kaftan"]),
    ("bottom", ["jeans", "pants", "trousers", "skirt", "skirts", "shorts", "leggings", "culottes",
                "joggers", "chinos", "bottoms"]),
    ("outerwear", ["jacket", "jackets", "coat", "coats", "blazer", "parka", "trench", "puffer",
                   "cardigan", "outerwear", "anorak", "bomber", "gilet", "cape"]),
    ("bag", ["bag", "bags", "handbag", "purse", "clutch", "tote", "backpack", "satchel",
             "crossbody", "shoulder bag", "hobo"]),
    ("shoes", ["shoe", "shoes", "sneaker", "sneakers", "boot", "boots", "booties", "heel", "heels",
               "pump", "pumps", "sandal", "sandals", "loafer", "loafers", "flats", "mule", "mules",
               "espadrille", "espadrilles", "slipper", "oxford", "oxfords", "wedges", "trainers",
               "footwear"]),
    ("hat", ["hat", "hats", "cap", "caps", "beanie", "beret", "fedora", "bucket hat"]),
    ("accessory", ["necklace", "earring", "earrings", "bracelet", "ring", "rings", "jewelry",
                   "jewellery", "watch", "watches", "sunglasses", "eyewear", "scarf", "scarves",
                   "belt", "belts", "gloves", "brooch", "hair", "headband", "wallet", "pendant",
                   "accessories"]),
    ("top", ["top", "tops", "tee", "t-shirt", "tshirt", "shirt", "shirts", "blouse", "blouses",
             "sweater", "sweaters", "sweatshirt", "hoodie", "tank", "camisole", "cami",
             "bodysuit", "pullover", "knit", "knitwear", "tunic", "polo", "jumper", "turtleneck",
             "vest"]),
]

_COMPILED: list[tuple[str, re.Pattern[str]]] = [
    (group, re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE))
    for group, words in _KEYWORDS
]


def group_from_text(*texts: str | None) -> str | None:
    """Return the catalog group implied by category/title text, or None.

    Texts are tried in order (pass the category first, the title second); the
    first text that matches any rule decides.
    """
    for text in texts:
        if not text or not isinstance(text, str):
            continue
        for group, pattern in _COMPILED:
            if pattern.search(text):
                return group
    return None


# --------------------------------------------------------------------------
# Garment kinds (from the segmentation model)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GarmentKind:
    name: str  # display name
    seg_labels: tuple[int, ...]  # label ids in the segmentation map
    search_groups: tuple[str, ...]  # catalog groups this garment may match
    min_area: float  # min fraction of image pixels to count as detected
    color: tuple[int, int, int]  # overlay colour


# Label ids of mattmdjaga/segformer_b2_clothes (ATR-style 18 classes):
# 0 Background, 1 Hat, 2 Hair, 3 Sunglasses, 4 Upper-clothes, 5 Skirt, 6 Pants,
# 7 Dress, 8 Belt, 9 Left-shoe, 10 Right-shoe, 11 Face, 12 Left-leg,
# 13 Right-leg, 14 Left-arm, 15 Right-arm, 16 Bag, 17 Scarf
GARMENT_KINDS: tuple[GarmentKind, ...] = (
    GarmentKind("Top", (4,), ("top", "outerwear"), 0.010, (222, 120, 96)),
    GarmentKind("Dress", (7,), ("dress",), 0.015, (176, 112, 160)),
    GarmentKind("Skirt", (5,), ("bottom",), 0.008, (120, 150, 210)),
    GarmentKind("Pants", (6,), ("bottom",), 0.010, (90, 140, 200)),
    GarmentKind("Shoes", (9, 10), ("shoes",), 0.002, (96, 170, 130)),
    GarmentKind("Bag", (16,), ("bag",), 0.004, (214, 170, 80)),
    GarmentKind("Hat", (1,), ("hat",), 0.003, (150, 120, 90)),
    GarmentKind("Scarf", (17,), ("accessory",), 0.004, (200, 100, 130)),
    GarmentKind("Sunglasses", (3,), ("accessory",), 0.0015, (80, 80, 80)),
    GarmentKind("Belt", (8,), ("accessory",), 0.0015, (130, 100, 70)),
)

KIND_BY_NAME: dict[str, GarmentKind] = {k.name: k for k in GARMENT_KINDS}


def complete_look_targets(present: set[str] | list[str], max_groups: int = 4) -> list[str]:
    """Which catalog groups to suggest to finish an outfit.

    ``present`` holds the primary catalog group of each detected garment. A
    detected top covers "top" only: outerwear is still worth suggesting as a
    layer.
    """
    present = set(present)
    targets: list[str] = []
    if "dress" not in present:
        if "top" not in present and "outerwear" not in present:
            targets.append("top")
        if "bottom" not in present:
            targets.append("bottom")
    for group in ("shoes", "bag", "outerwear", "accessory"):
        if group not in present:
            targets.append(group)
    return targets[:max_groups]
