from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import load_and_preprocess_image, load_uploaded_image


def _exif_rotated_jpeg() -> bytes:
    # EXIF orientation 6 means the stored pixels need a 90 degree clockwise
    # rotation to match the camera's intended portrait orientation.
    image = Image.new("RGB", (80, 120), (140, 80, 35))
    exif = Image.Exif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def test_index_and_upload_loaders_apply_exif_orientation(tmp_path):
    encoded = _exif_rotated_jpeg()
    image_path = tmp_path / "portrait.jpg"
    image_path.write_bytes(encoded)

    indexed = load_and_preprocess_image(image_path)
    uploaded = load_uploaded_image(BytesIO(encoded))

    assert indexed is not None
    assert uploaded is not None
    assert indexed.shape == (80, 120, 3)
    assert uploaded.size == (120, 80)
    np.testing.assert_array_equal(indexed, np.asarray(uploaded))
