import numpy as np
import cv2
import pytest

from app.vision.board_detector import (
    DetectedBoard,
    detect_board,
    order_corners,
    perspective_transform,
    split_into_squares,
    get_square_name,
    BOARD_OUTPUT_SIZE,
    SQUARE_SIZE,
)


def create_synthetic_board(size: int = 800) -> np.ndarray:
    """Create a synthetic 8x8 chessboard image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // 8
    for row in range(8):
        for col in range(8):
            x1 = col * sq
            y1 = row * sq
            x2 = x1 + sq
            y2 = y1 + sq
            if (row + col) % 2 == 0:
                color = (255, 255, 255)
            else:
                color = (50, 50, 50)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img


class TestGetSquareName:
    def test_a8(self) -> None:
        assert get_square_name(0) == "a8"

    def test_h8(self) -> None:
        assert get_square_name(7) == "h8"

    def test_a1(self) -> None:
        assert get_square_name(56) == "a1"

    def test_h1(self) -> None:
        assert get_square_name(63) == "h1"

    def test_e4(self) -> None:
        # e4 = col 4, row 4 from top -> index 4*8 + 4 = 36
        assert get_square_name(36) == "e4"

    def test_all_squares_named(self) -> None:
        names = [get_square_name(i) for i in range(64)]
        assert len(names) == 64
        assert len(set(names)) == 64


class TestOrderCorners:
    def test_already_ordered(self) -> None:
        pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        ordered = order_corners(pts)
        assert ordered[0].tolist() == [0, 0]
        assert ordered[1].tolist() == [100, 0]
        assert ordered[2].tolist() == [100, 100]
        assert ordered[3].tolist() == [0, 100]

    def test_shuffled(self) -> None:
        pts = np.array([[100, 100], [0, 0], [0, 100], [100, 0]], dtype=np.float32)
        ordered = order_corners(pts)
        assert ordered[0].tolist() == [0, 0]
        assert ordered[1].tolist() == [100, 0]
        assert ordered[2].tolist() == [100, 100]
        assert ordered[3].tolist() == [0, 100]


class TestSplitIntoSquares:
    def test_returns_64_squares(self) -> None:
        board = create_synthetic_board(800)
        squares = split_into_squares(board)
        assert len(squares) == 64

    def test_square_size(self) -> None:
        board = create_synthetic_board(800)
        squares = split_into_squares(board)
        for sq in squares:
            assert sq.shape[0] == SQUARE_SIZE
            assert sq.shape[1] == SQUARE_SIZE

    def test_squares_are_different(self) -> None:
        board = create_synthetic_board(800)
        squares = split_into_squares(board)
        # Check that at least some squares are visually different
        # (white vs dark squares)
        assert not np.array_equal(squares[0], squares[1])


class TestPerspectiveTransform:
    def test_transform_returns_image(self) -> None:
        img = create_synthetic_board(800)
        corners = np.array([
            [0, 0], [800, 0], [800, 800], [0, 800]
        ], dtype=np.float32)
        result = perspective_transform(img, corners)
        assert result is not None
        assert result.shape[0] == BOARD_OUTPUT_SIZE
        assert result.shape[1] == BOARD_OUTPUT_SIZE

    def test_transform_with_offset(self) -> None:
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (900, 900), (255, 255, 255), -1)
        corners = np.array([
            [100, 100], [900, 100], [900, 900], [100, 900]
        ], dtype=np.float32)
        result = perspective_transform(img, corners)
        assert result is not None


class TestDetectBoard:
    def test_detect_synthetic_board(self) -> None:
        img = create_synthetic_board(800)
        result = detect_board(img)
        # Fallback should detect the board
        assert result is not None
        assert result.transformed.shape[0] == BOARD_OUTPUT_SIZE
        assert len(result.squares) == 64

    def test_detect_returns_none_for_too_small(self) -> None:
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        result = detect_board(img)
        assert result is None

    def test_detect_none_input(self) -> None:
        result = detect_board(None)
        assert result is None

    def test_detect_empty_input(self) -> None:
        result = detect_board(np.array([]))
        assert result is None

    def test_detected_board_is_valid(self) -> None:
        img = create_synthetic_board(800)
        result = detect_board(img)
        assert result is not None
        assert result.is_valid

    def test_fallback_grid_detection(self) -> None:
        """A mostly uniform image should use fallback and still detect."""
        img = np.ones((600, 600, 3), dtype=np.uint8) * 128
        result = detect_board(img)
        # Fallback should handle this
        assert result is not None
        assert len(result.squares) == 64


class TestGeometryReuse:
    def test_detection_produces_reusable_geometry(self) -> None:
        img = create_synthetic_board(800)
        board = detect_board(img)
        assert board is not None
        assert board.corners.shape == (4, 2)
        assert len(board.square_names) == 64

    def test_reuse_generates_same_squares(self) -> None:
        img = create_synthetic_board(800)
        board = detect_board(img)
        assert board is not None
        reused = board.split_capture(img)
        assert len(reused) == 64
        for original, new in zip(board.squares, reused):
            assert original.shape == new.shape

    def test_reuses_geometry_on_second_capture(self) -> None:
        img = create_synthetic_board(800)
        board = detect_board(img)
        assert board is not None

        # A second, slightly different capture of the same region.
        img2 = create_synthetic_board(800)
        cv2.rectangle(
            img2, (100, 100), (200, 200), (0, 0, 255), -1
        )

        squares2 = board.split_capture(img2)
        normal2 = board.transform_capture(img2)
        assert normal2 is not None
        assert normal2.shape[0] == BOARD_OUTPUT_SIZE
        assert normal2.shape[1] == BOARD_OUTPUT_SIZE
        assert len(squares2) == 64
        for sq in squares2:
            assert sq.shape[0] == SQUARE_SIZE
            assert sq.shape[1] == SQUARE_SIZE

        # The two captures differ, so at least one square differs.
        assert not all(
            np.array_equal(a, b)
            for a, b in zip(board.squares, squares2)
        )

    def test_reuse_does_not_run_detection(self, monkeypatch) -> None:
        img = create_synthetic_board(800)
        board = detect_board(img)
        assert board is not None

        called = []

        def fail_if_called(*args, **kwargs):
            called.append(True)
            raise AssertionError("detect_board must not run during reuse")

        monkeypatch.setattr(
            "app.vision.board_detector.detect_board", fail_if_called
        )

        img2 = create_synthetic_board(800)
        normalized = board.transform_capture(img2)
        squares2 = board.split_capture(img2)

        assert normalized is not None
        assert len(squares2) == 64
        assert called == []

    def test_square_ordering_is_deterministic(self) -> None:
        img = create_synthetic_board(800)
        board = detect_board(img)
        assert board is not None
        assert board.square_names[0] == "a8"
        assert board.square_names[7] == "h8"
        assert board.square_names[56] == "a1"
        assert board.square_names[63] == "h1"
        assert board.square_names == [
            get_square_name(i) for i in range(64)
        ]

    def test_split_capture_invalid_image(self) -> None:
        img = create_synthetic_board(800)
        board = detect_board(img)
        assert board is not None
        assert board.split_capture(None) == []
        assert board.transform_capture(np.array([])) is None
