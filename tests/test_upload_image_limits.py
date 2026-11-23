"""Query uploads are decoded in memory with bounded size and pixel counts."""
from __future__ import annotations

from io import BytesIO
import struct
import zlib

from PIL import Image

from utils import load_uploaded_image


class UploadedImage(BytesIO):
    def __init__(self, value: bytes):
        super().__init__(value)
        self.size = len(value)


def test_upload_decoder_accepts_supported_image_in_memory():
    buffer = BytesIO()
    Image.new("RGB", (20, 20), (40, 80, 120)).save(buffer, format="PNG")

    image = load_uploaded_image(UploadedImage(buffer.getvalue()))

    assert image is not None
    assert image.mode == "RGB"
    assert image.size == (20, 20)


def test_upload_decoder_rejects_oversized_payload():
    upload = UploadedImage(b"not decoded")
    upload.size = 25 * 1024 * 1024 + 1

    assert load_uploaded_image(upload) is None


def test_upload_decoder_rejects_extreme_pixel_dimensions_without_loading():
    header = b"IHDR" + struct.pack(">IIBBBBB", 10001, 4000, 8, 2, 0, 0, 0)
    chunk = struct.pack(">I", len(header) - 4) + header + struct.pack(">I", zlib.crc32(header) & 0xFFFFFFFF)
    png_header = b"\x89PNG\r\n\x1a\n" + chunk

    assert load_uploaded_image(UploadedImage(png_header)) is None
