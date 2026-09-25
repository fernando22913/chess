import os
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.capture.screen_capture import (
    ScreenRegion,
    capture_region,
    capture_region_pil,
    get_screen_size,
    MIN_REGION_SIZE,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestScreenRegion:
    def test_valid_region(self) -> None:
        r = ScreenRegion(0, 0, 800, 600)
        assert r.is_valid()

    def test_too_small_width(self) -> None:
        r = ScreenRegion(0, 0, 10, 600)
        assert not r.is_valid()

    def test_too_small_height(self) -> None:
        r = ScreenRegion(0, 0, 800, 10)
        assert not r.is_valid()

    def test_both_too_small(self) -> None:
        r = ScreenRegion(0, 0, 10, 10)
        assert not r.is_valid()

    def test_exactly_minimum(self) -> None:
        r = ScreenRegion(0, 0, MIN_REGION_SIZE, MIN_REGION_SIZE)
        assert r.is_valid()

    def test_mss_dict(self) -> None:
        r = ScreenRegion(100, 200, 800, 600)
        d = r.to_mss_dict()
        assert d == {"left": 100, "top": 200, "width": 800, "height": 600}


class TestCaptureRegion:
    def test_capture_returns_ndarray(self) -> None:
        screen_w, screen_h = get_screen_size()
        r = ScreenRegion(0, 0, min(100, screen_w), min(100, screen_h))
        img = capture_region(r)
        assert isinstance(img, np.ndarray)
        assert img.shape[0] > 0
        assert img.shape[1] > 0
        assert img.shape[2] == 3  # BGR

    def test_capture_too_small_raises(self) -> None:
        r = ScreenRegion(0, 0, 10, 10)
        with pytest.raises(ValueError, match="too small"):
            capture_region(r)

    def test_capture_pil_returns_image(self) -> None:
        from PIL import Image
        screen_w, screen_h = get_screen_size()
        r = ScreenRegion(0, 0, min(100, screen_w), min(100, screen_h))
        img = capture_region_pil(r)
        assert isinstance(img, Image.Image)
        assert img.size[0] > 0
        assert img.size[1] > 0


class TestScreenSize:
    def test_get_screen_size(self) -> None:
        w, h = get_screen_size()
        assert w > 0
        assert h > 0
