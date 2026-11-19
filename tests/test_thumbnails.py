import io
from pathlib import Path

import pytest
from PIL import Image

from utils import (
    clamp_face_box,
    encode_face_thumbnail,
    encode_image_thumbnail,
    parse_face_location,
)


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (400, 300), (200, 120, 90)).save(path, format="JPEG")
    return path


def _size(data: bytes):
    return Image.open(io.BytesIO(data)).size


def test_parse_face_location_accepts_all_shapes():
    assert parse_face_location("10,20,30,40") == (10, 20, 30, 40)
    assert parse_face_location((1, 2, 3, 4)) == (1, 2, 3, 4)
    assert parse_face_location([5, 6, 7, 8]) == (5, 6, 7, 8)
    assert parse_face_location({"top": 1, "right": 2, "bottom": 3, "left": 4}) == (1, 2, 3, 4)


def test_parse_face_location_rejects_garbage():
    assert parse_face_location("garbage") is None
    assert parse_face_location(None) is None
    assert parse_face_location("1,2,3") is None


def test_clamp_face_box_respects_bounds():
    assert clamp_face_box((10, 50, 60, 20), 100, 100, padding=0) == (10, 50, 60, 20)
    assert clamp_face_box((-20, 200, 200, -20), 100, 100, padding=10) == (0, 100, 100, 0)


def test_face_thumbnail_is_small_jpeg(sample_image):
    data = encode_face_thumbnail(sample_image, "50,250,200,50", size=(64, 64))
    assert data is not None
    assert data[:2] == b"\xff\xd8"  # JPEG magic bytes
    assert max(_size(data)) <= 64


def test_face_thumbnail_falls_back_to_whole_image(sample_image):
    data = encode_face_thumbnail(sample_image, None, size=(32, 32))
    assert data is not None
    assert max(_size(data)) <= 32


def test_face_thumbnail_missing_file_returns_none(tmp_path):
    assert encode_face_thumbnail(tmp_path / "missing.jpg", "0,10,10,0") is None


def test_image_thumbnail_keeps_aspect_ratio(sample_image):
    data = encode_image_thumbnail(sample_image, size=(100, 100))
    assert data is not None
    assert _size(data) == (100, 75)
