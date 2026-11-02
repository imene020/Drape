import numpy as np

from drape.garments import extract_garments, make_crop, render_overlay, resize_max_side


def test_extracts_expected_pieces(outfit_label_map):
    names = [g.name for g in extract_garments(outfit_label_map)]
    assert names == ["Top", "Pants", "Shoes", "Bag"]  # hair ignored, shoes merged


def test_bbox_and_area(outfit_label_map):
    top = next(g for g in extract_garments(outfit_label_map) if g.name == "Top")
    assert top.bbox == (15, 10, 45, 45)
    assert abs(top.area_frac - (35 * 30) / (100 * 60)) < 1e-9


def test_tiny_detections_are_dropped():
    lm = np.zeros((200, 200), dtype=np.uint8)
    lm[0:3, 0:3] = 16  # 9 px bag: noise
    assert extract_garments(lm) == []


def test_speckles_removed_from_mask():
    lm = np.zeros((200, 200), dtype=np.uint8)
    lm[50:150, 50:150] = 4
    lm[5:7, 190:192] = 4  # far-away speckle
    top = extract_garments(lm)[0]
    assert top.bbox == (50, 50, 150, 150)


def test_split_dress_prefers_top_and_skirt_when_they_dominate():
    lm = np.zeros((100, 100), dtype=np.uint8)
    lm[10:50, 20:80] = 4
    lm[50:90, 20:80] = 5
    lm[48:52, 40:60] = 7  # small dress fragment
    names = [g.name for g in extract_garments(lm)]
    assert "Dress" not in names and {"Top", "Skirt"} <= set(names)


def test_crop_is_square_white_background(outfit_image, outfit_label_map):
    top = extract_garments(outfit_label_map)[0]
    crop = make_crop(outfit_image, top, min_side=8)
    assert crop.width == crop.height
    arr = np.asarray(crop)
    assert tuple(arr[0, 0]) == (255, 255, 255)
    centre = arr[arr.shape[0] // 2, arr.shape[1] // 2]
    assert centre[0] > 150 and centre[2] < 80  # red garment survives


def test_overlay_keeps_size(outfit_image, outfit_label_map):
    out = render_overlay(outfit_image, extract_garments(outfit_label_map))
    assert out.size == outfit_image.size


def test_resize_max_side():
    from PIL import Image

    im = Image.new("RGB", (4000, 2000))
    assert resize_max_side(im, 1000).size == (1000, 500)
    assert resize_max_side(Image.new("RGB", (300, 200)), 1000).size == (300, 200)
