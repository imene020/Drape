import numpy as np
import pandas as pd
import pytest

from drape.index import CatalogIndex, mmr, zscore
from fakes import product


def test_finds_same_colour_within_group(catalog, embedder):
    q = embedder.embed_images([product((205, 35, 25))])[0]
    res = catalog.search(q, k=3, groups=("shoes",))
    assert res[0].item_id == "shoes-red"
    assert all(m.group == "shoes" for m in res)
    assert 0.9 < res[0].similarity <= 1.0001


def test_group_filter_union(catalog, embedder):
    q = embedder.embed_images([product((30, 60, 200))])[0]
    res = catalog.search(q, k=6, groups=("top", "outerwear"), diversify=False)
    assert {m.group for m in res} == {"top", "outerwear"}


def test_unknown_group_returns_nothing(catalog, embedder):
    q = embedder.embed_images([product((30, 60, 200))])[0]
    assert catalog.search(q, groups=("spaceships",)) == []


def test_exclude(catalog, embedder):
    q = embedder.embed_images([product((200, 30, 30))])[0]
    first = catalog.search(q, k=1, groups=("bag",))[0]
    again = catalog.search(q, k=1, groups=("bag",), exclude=[first.row])[0]
    assert again.row != first.row


def test_text_fusion_can_change_ranking(catalog, embedder):
    q = embedder.embed_images([product((200, 30, 30))])[0]
    t = embedder.embed_texts(["anything"])[0]
    base = [m.row for m in catalog.search(q, k=3, groups=("dress",), diversify=False)]
    heavy_text = [m.row for m in catalog.search(q, k=3, groups=("dress",), text_query=t, alpha=1.0, diversify=False)]
    expected = list(catalog.rows_for(("dress",))[np.argsort(-(catalog.embeddings[catalog.rows_for(("dress",))] @ t))])
    assert heavy_text == expected
    assert base[0] == catalog.meta.index[catalog.meta.item_id == "dress-red"][0]


def test_text_search_has_no_fake_percentages(catalog, embedder):
    t = embedder.embed_texts(["red dress"])[0]
    res = catalog.search_text(t, k=4)
    assert len(res) == 4 and all(m.similarity is None for m in res)
    assert "%" not in res[0].caption()


def test_mmr_no_duplicates():
    emb = np.eye(5, dtype=np.float32)
    emb = np.vstack([emb, emb[0]])  # row 5 duplicates row 0
    picked = mmr(np.arange(6), np.array([1.0, 0.2, 0.2, 0.2, 0.2, 0.99]), emb, k=3)
    assert len(set(picked)) == 3 and not {0, 5} <= set(picked)


def test_zscore_constant():
    assert np.allclose(zscore(np.ones(4)), 0)


def test_save_load_roundtrip(catalog, tmp_path):
    catalog.save(tmp_path)
    loaded = CatalogIndex.load(tmp_path)
    assert len(loaded) == len(catalog) and loaded.dim == catalog.dim
    assert np.allclose(loaded.embeddings, catalog.embeddings, atol=2e-3)
    assert loaded.manifest["embed_model"] == "fake"
    assert loaded.thumbnail(0).size == (64, 64)


def test_rejects_mismatched_rows():
    with pytest.raises(ValueError):
        CatalogIndex(np.zeros((2, 4)), pd.DataFrame({"item_id": ["a"], "title": ["a"], "group": ["top"]}))


def test_caption_truncates():
    from drape.index import Match

    m = Match(0, "x", "a" * 80, "top", 0.9, 0.876)
    assert m.caption().endswith("88% similar") and "…" in m.caption()
