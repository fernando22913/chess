from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


BOARD_OUTPUT_SIZE = 800
SQUARE_SIZE = BOARD_OUTPUT_SIZE // 8


@dataclass
class DetectedBoard:
    corners: np.ndarray
    transformed: np.ndarray
    squares: list[np.ndarray]

    @property
    def is_valid(self) -> bool:
        return self.transformed.shape[0] > 0 and self.transformed.shape[1] > 0

    @property
    def square_names(self) -> list[str]:
        """The 64 algebraic names (a8..h1) in the same order as ``squares``."""
        return [get_square_name(i) for i in range(64)]

    def transform_capture(self, image: np.ndarray) -> np.ndarray | None:
        """Normalize a new capture of the same region using stored geometry.

        Reuses the already-detected corners, so board detection is NOT rerun.
        """
        if image is None or image.size == 0:
            return None
        return perspective_transform(image, self.corners)

    def split_capture(self, image: np.ndarray) -> list[np.ndarray]:
        """Return the 64 squares of a new capture using stored geometry.

        Uses the stored corners only; board detection is never rerun.
        """
        normalized = self.transform_capture(image)
        if normalized is None:
            return []
        return split_into_squares(normalized)


def detect_board(image: np.ndarray) -> DetectedBoard | None:
    """Detect chessboard in a BGR image. Returns DetectedBoard or None."""
    if image is None or image.size == 0:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_grid_detection(image)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 1000:
            corners = order_corners(approx.reshape(4, 2))
            transformed = perspective_transform(image, corners)
            if transformed is not None:
                squares = split_into_squares(transformed)
                return DetectedBoard(
                    corners=corners,
                    transformed=transformed,
                    squares=squares,
                )

    return _fallback_grid_detection(image)


def _fallback_grid_detection(image: np.ndarray) -> DetectedBoard | None:
    """Fallback: assume the board fills most of the image."""
    h, w = image.shape[:2]
    size = min(h, w)
    margin_x = (w - size) // 2
    margin_y = (h - size) // 2

    if size < 100:
        return None

    corners = np.array([
        [margin_x, margin_y],
        [margin_x + size, margin_y],
        [margin_x + size, margin_y + size],
        [margin_x, margin_y + size],
    ], dtype=np.float32)

    transformed = perspective_transform(image, corners)
    if transformed is None:
        return None

    squares = split_into_squares(transformed)
    return DetectedBoard(
        corners=corners,
        transformed=transformed,
        squares=squares,
    )


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def perspective_transform(
    image: np.ndarray, corners: np.ndarray, output_size: int = BOARD_OUTPUT_SIZE
) -> np.ndarray | None:
    """Apply perspective transform to get a flat square board."""
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (output_size, output_size))

    if warped.size == 0:
        return None

    return warped


def split_into_squares(board_image: np.ndarray) -> list[np.ndarray]:
    """Split an 800x800 board image into 64 squares (8x8 grid)."""
    squares = []
    for row in range(8):
        for col in range(8):
            y1 = row * SQUARE_SIZE
            y2 = y1 + SQUARE_SIZE
            x1 = col * SQUARE_SIZE
            x2 = x1 + SQUARE_SIZE
            squares.append(board_image[y1:y2, x1:x2])
    return squares


def get_square_name(index: int) -> str:
    """Convert square index (0-63) to algebraic name (a8, b8, ..., h1)."""
    row = index // 8
    col = index % 8
    file_letter = chr(ord("a") + col)
    rank_number = 8 - row
    return f"{file_letter}{rank_number}"
