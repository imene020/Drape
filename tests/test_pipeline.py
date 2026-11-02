import numpy as np
from PIL import Image

from drape.pipeline import ShopTheLook
from fakes import FakeSegmenter, product


def make_engine(catalog, embedder, label_map):
    return ShopTheLook(embedder, FakeSegmenter(label_map), catalog, max_side=512)


def test_outfit_mode_matches_each_piece_in_its_category(catalog, embedder, outfit_image, outfit_label_map):
    res = make_engine(catalog, embedder, outfit_label_map).analyze(outfit_image, k=3)
    assert res.mode == "outfit"
    assert res.kind_names == ["Top", "Pants", "Shoes", "Bag"]
    by_name = {g.name: g for g in res.garments}
    assert by_name["Top"].matches[0].item_id in {"top-red", "outerwear-red"}
    assert by_name["Pants"].matches[0].item_id == "bottom-blue"
    assert by_name["Shoes"].matches[0].item_id == "shoes-green"
    assert by_name["Bag"].matches[0].item_id == "bag-red"
    for g in res.garments:
        assert all(m.group in g.kind.search_groups for m in g.matches)
    assert res.overlay.size == res.image.size
    assert set(res.timings) == {"segment_s", "embed_s", "search_s", "total_s"}


def test_item_mode_when_no_person(catalog, embedder):
    blank = np.zeros((64, 64), dtype=np.uint8)
    res = make_engine(catalog, embedder, blank).analyze(product((30, 160, 60)), k=3)
    assert res.mode == "item" and len(res.garments) == 1
    g = res.garments[0]
    assert g.matches and all(m.group in g.kind.search_groups for m in g.matches)


def test_complete_look_suggests_only_missing_categories(catalog, embedder, outfit_image, outfit_label_map):
    engine = make_engine(catalog, embedder, outfit_label_map)
    res = engine.analyze(outfit_image)
    looks = engine.complete_look(res.look_embedding, res.present_groups, k_per_group=2)
    assert set(looks) == {"outerwear", "accessory"}  # top, bottom, shoes, bag already worn
    assert all(len(v) == 2 for v in looks.values())


def test_style_text_is_used(catalog, embedder, outfit_image, outfit_label_map):
    engine = make_engine(catalog, embedder, outfit_label_map)
    res = engine.analyze(outfit_image)
    looks = engine.complete_look(res.look_embedding, ["dress"], k_per_group=3, style_text="evening")
    assert "shoes" in looks and "top" not in looks


def test_refine_without_text_equals_search(catalog, embedder, outfit_image, outfit_label_map):
    engine = make_engine(catalog, embedder, outfit_label_map)
    g = engine.analyze(outfit_image).garments[0]
    plain = engine.refine(g.embedding, g.kind.search_groups, "", k=4)
    assert [m.row for m in plain] == [m.row for m in catalog.search(g.embedding, k=4, groups=g.kind.search_groups)]
    twisted = engine.refine(g.embedding, g.kind.search_groups, "in black", alpha=0.9, k=4)
    assert all(m.group in g.kind.search_groups for m in twisted)


def test_text_search(catalog, embedder):
    engine = make_engine(catalog, embedder, np.zeros((8, 8), np.uint8))
    assert engine.text_search("   ") == []
    res = engine.text_search("red shoes", k=3, group="shoes")
    assert len(res) == 3 and all(m.group == "shoes" for m in res)


def test_large_photo_is_downscaled(catalog, embedder, outfit_label_map):
    big = Image.new("RGB", (3000, 5000), (240, 240, 240))
    res = make_engine(catalog, embedder, outfit_label_map).analyze(big)
    assert max(res.image.size) == 512
