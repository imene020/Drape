import pytest

from drape.taxonomy import GARMENT_KINDS, GROUPS, complete_look_targets, group_from_text


@pytest.mark.parametrize(
    "texts, expected",
    [
        (("Day Dresses",), "dress"),
        (("Shirt dress in cotton",), "dress"),
        (("Denim jacket",), "outerwear"),
        (("Top-handle bag",), "bag"),
        (("Skinny jeans",), "bottom"),
        (("Bootcut jeans",), "bottom"),
        (("Ankle boots",), "shoes"),
        (("Gold hoop earrings",), "accessory"),
        (("Crop top",), "top"),
        (("Matte lipstick",), "other"),
        (("Wool beanie",), "hat"),
        ((None, "Leather sneakers"), "shoes"),
        (("", "random words"), None),
    ],
)
def test_group_from_text(texts, expected):
    assert group_from_text(*texts) == expected


def test_category_text_beats_title():
    assert group_from_text("Skirts", "goes great with a crop top") == "bottom"


def test_word_boundaries():
    # "ring" must not match "earring" as a ring, "cap" must not match "cape" etc.
    assert group_from_text("String bikini") is None
    assert group_from_text("Baggy sweater") == "top"


def test_every_garment_maps_to_known_groups():
    for kind in GARMENT_KINDS:
        assert kind.search_groups and set(kind.search_groups) <= set(GROUPS)


def test_complete_look_targets():
    assert complete_look_targets(["top", "bottom"]) == ["shoes", "bag", "outerwear", "accessory"]
    assert complete_look_targets(["dress"])[:2] == ["shoes", "bag"]
    assert "top" not in complete_look_targets(["dress"])
    assert complete_look_targets(["shoes"])[:2] == ["top", "bottom"]
    assert "top" not in complete_look_targets(["outerwear"])
    assert len(complete_look_targets([], max_groups=3)) == 3
