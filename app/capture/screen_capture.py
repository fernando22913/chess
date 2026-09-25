from __future__ import annotations

from dataclasses import dataclass

import mss
import numpy as np
from PIL import Image


MIN_REGION_SIZE = 50


@dataclass
class ScreenRegion:
    x: int
    y: int
    width: int
    height: int

    def is_valid(self) -> bool:
        return (
            self.width >= MIN_REGION_SIZE
            and self.height >= MIN_REGION_SIZE
        )

    def to_mss_dict(self) -> dict:
        return {"left": self.x, "top": self.y, "width": self.width, "height": self.height}


def capture_region(region: ScreenRegion) -> np.ndarray:
    """Capture a screen region and return it as a BGR numpy array."""
    if not region.is_valid():
        raise ValueError(
            f"Region too small: {region.width}x{region.height}. "
            f"Minimum is {MIN_REGION_SIZE}x{MIN_REGION_SIZE}."
        )

    with mss.MSS() as sct:
        screenshot = sct.grab(region.to_mss_dict())
        img = np.array(screenshot)

    # mss returns BGRA, convert to BGR for OpenCV compatibility
    return img[:, :, :3]


def capture_region_pil(region: ScreenRegion) -> Image.Image:
    """Capture a screen region and return it as a PIL Image."""
    if not region.is_valid():
        raise ValueError(
            f"Region too small: {region.width}x{region.height}. "
            f"Minimum is {MIN_REGION_SIZE}x{MIN_REGION_SIZE}."
        )

    with mss.MSS() as sct:
        screenshot = sct.grab(region.to_mss_dict())
        img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

    return img


def get_screen_size() -> tuple[int, int]:
    """Return (width, height) of the primary monitor."""
    with mss.MSS() as sct:
        monitor = sct.monitors[1]
        return monitor["width"], monitor["height"]
