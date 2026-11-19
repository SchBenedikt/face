"""
Cached, render-ready image helpers for the Streamlit UI.

Streamlit re-runs the whole script on every interaction (slider move, button
click, page switch). Without caching, every re-run decoded every full-resolution
photo again and rebuilt its thumbnail, which is what made the gallery and the
search results feel sluggish.

The cache key always contains the file's ``mtime``, so replacing an image on
disk invalidates its thumbnails automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import streamlit as st

from utils import (
    normalise_box_size,
    encode_face_thumbnail,
    encode_image_thumbnail,
    parse_face_location,
)

THUMB_CACHE_ENTRIES = 4000


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@st.cache_data(show_spinner=False, max_entries=THUMB_CACHE_ENTRIES)
def _face_thumbnail_cached(
    path: str,
    mtime: float,
    box: Tuple[int, int, int, int],
    size: Tuple[int, int],
    padding: int,
) -> Optional[bytes]:
    # ``mtime`` is only part of the cache key; the file itself is read from disk.
    return encode_face_thumbnail(path, box, size=size, padding=padding)


@st.cache_data(show_spinner=False, max_entries=THUMB_CACHE_ENTRIES)
def _image_thumbnail_cached(
    path: str,
    mtime: float,
    size: Tuple[int, int],
) -> Optional[bytes]:
    return encode_image_thumbnail(path, size=size)


def face_thumbnail(
    image_path: Union[str, Path],
    location=None,
    size: Union[int, Tuple[int, int]] = (240, 240),
    padding: int = 10,
) -> Optional[bytes]:
    """
    Return a cached JPEG thumbnail (bytes) for a face or a whole image.

    ``location`` accepts everything :func:`utils.parse_face_location` supports.
    When it is unusable, a thumbnail of the whole image is returned instead.
    """
    path = Path(str(image_path))
    if not path.exists():
        return None

    normalised_size = normalise_box_size(size)
    box = parse_face_location(location)
    if box is None:
        return _image_thumbnail_cached(str(path), _mtime(path), normalised_size)
    return _face_thumbnail_cached(
        str(path), _mtime(path), box, normalised_size, int(padding)
    )


def clear_thumbnail_cache() -> None:
    """Drop all cached thumbnails (used by the manual refresh buttons)."""
    st.cache_data.clear()
