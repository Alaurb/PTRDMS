from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Input image directory does not exist: {folder}")
    return sorted(
        (path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )


def _face_vectors(side: str, face_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, face_size, dtype=np.float32)
    u, image_v = np.meshgrid(axis, axis)
    y = -image_v

    if side == "left":
        x = -np.ones_like(u)
        z = u
    elif side == "right":
        x = np.ones_like(u)
        z = -u
    elif side == "front":
        x, z = u, np.ones_like(u)
    elif side == "back":
        x, z = -u, -np.ones_like(u)
    elif side == "top":
        x, y, z = u, np.ones_like(u), image_v
    elif side == "bottom":
        x, y, z = u, -np.ones_like(u), -image_v
    else:
        raise ValueError(f"Unknown cube face: {side}")

    norm = np.sqrt(x * x + y * y + z * z)
    return x / norm, y / norm, z / norm


def _bilinear_remap(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = np.floor(map_x).astype(np.int32) % width
    y0 = np.clip(np.floor(map_y).astype(np.int32), 0, height - 1)
    x1 = (x0 + 1) % width
    y1 = np.clip(y0 + 1, 0, height - 1)

    wx = (map_x - np.floor(map_x))[..., None]
    wy = (map_y - np.floor(map_y))[..., None]
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return np.clip(top * (1.0 - wy) + bottom * wy, 0, 255).astype(np.uint8)


def extract_cube_face(panorama: Image.Image, side: str, face_size: int = 720) -> Image.Image:
    """Project an equirectangular panorama to a 90-degree cubemap face."""
    if face_size < 64:
        raise ValueError("face_size must be at least 64 pixels")
    source = np.asarray(panorama.convert("RGB"), dtype=np.float32)
    height, width = source.shape[:2]
    x, y, z = _face_vectors(side, face_size)
    longitude = np.arctan2(x, z)
    latitude = np.arcsin(y)
    map_x = (longitude + np.pi) / (2.0 * np.pi) * width
    map_y = (0.5 - latitude / np.pi) * height
    return Image.fromarray(_bilinear_remap(source, map_x, map_y), mode="RGB")


def extract_side_views(panorama: Image.Image, face_size: int = 720) -> dict[str, Image.Image]:
    return {
        "left": extract_cube_face(panorama, "left", face_size),
        "right": extract_cube_face(panorama, "right", face_size),
    }


def extract_all_faces(panorama: Image.Image, face_size: int = 720) -> dict[str, Image.Image]:
    """Six computed perspective projections, not crops or camera-supplied faces."""
    return {side: extract_cube_face(panorama, side, face_size)
            for side in ("front", "right", "back", "left", "top", "bottom")}
