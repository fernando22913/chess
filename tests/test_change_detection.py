"""Tests for the cheap change gate and the watch loop."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.capture.board_watcher import (
    DEFAULT_INTERVAL_MS,
    MIN_INTERVAL_MS,
    BoardWatcher,
    WatchConfig,
)
from app.capture.screen_capture import ScreenRegion
from app.chess.position import positions_equal
from app.chess.validation import (
    INVALID_POSITION_MESSAGE,
    LOW_CONFIDENCE_HEADER,
    TEMPORARY_VISUAL_MESSAGE,
)
from app.vision.board_detector import (
    DetectedBoard,
    perspective_transform,
    split_into_squares,
)
from app.vision.change_detector import (
    DEFAULT_CHANGE_RATIO,
    SIGNATURE_SIZE,
    board_signature,
    changed_ratio,
    has_changed,
)
from app.vision.piece_recognizer import PieceRecognizerError

REGION_SIZE = 400
START_FEN = "8/8/8/8/8/8/8/4K2k w - - 0 1"
MOVE_FEN = "8/8/8/4P3/8/8/8/4K2k w - - 0 1"


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def make_frame(block: tuple[int, int, int, int] | None = None, noise: int = 0) -> np.ndarray:
    """Synthetic screen region: flat background, optional bright block/noise."""
    frame = np.full((REGION_SIZE, REGION_SIZE, 3), 40, dtype=np.uint8)
    if block is not None:
        x, y, w, h = block
        frame[y:y + h, x:x + w] = 230
    if noise:
        frame[10:10 + noise, 10:10 + noise] = 255
    return frame


def make_board(frame: np.ndarray) -> DetectedBoard:
    """Board whose stored corners cover the whole region (no detection needed)."""
    corners = np.array(
        [
            [0, 0],
            [REGION_SIZE - 1, 0],
            [REGION_SIZE - 1, REGION_SIZE - 1],
            [0, REGION_SIZE - 1],
        ],
        dtype=np.float32,
    )
    transformed = perspective_transform(frame, corners)
    return DetectedBoard(
        corners=corners,
        transformed=transformed,
        squares=split_into_squares(transformed),
    )


def fen_to_squares(fen: str) -> list[str]:
    import chess

    board = chess.Board(fen)
    squares = []
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            squares.append(piece.symbol() if piece else ".")
    return squares


class FakeRecognizer:
    """Deterministic stand-in for the YOLO model."""

    def __init__(
        self,
        squares: list[str] | None = None,
        error: Exception | None = None,
        confidences: list[float] | None = None,
    ):
        self.squares = squares if squares is not None else fen_to_squares(START_FEN)
        self.error = error
        self.confidences = confidences
        self.calls = 0

    def recognize(self, image: np.ndarray) -> list[str]:
        return self.recognize_detailed(image)[0]

    def recognize_detailed(self, image: np.ndarray) -> tuple[list[str], list[float]]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        confidences = self.confidences
        if confidences is None:
            confidences = [0.95 if s != "." else 0.0 for s in self.squares]
        return list(self.squares), list(confidences)


class TestBoardSignature:
    def test_resizes_to_signature_size(self) -> None:
        signature = board_signature(np.zeros((800, 800, 3), dtype=np.uint8))
        assert signature.shape == (SIGNATURE_SIZE, SIGNATURE_SIZE)

    def test_accepts_grayscale(self) -> None:
        signature = board_signature(np.zeros((400, 400), dtype=np.uint8))
        assert signature.shape == (SIGNATURE_SIZE, SIGNATURE_SIZE)

    def test_rejects_empty_image(self) -> None:
        assert board_signature(None) is None
        assert board_signature(np.array([])) is None


class TestHasChanged:
    def test_identical_frames_have_no_change(self) -> None:
        a = board_signature(make_frame())
        b = board_signature(make_frame())
        assert changed_ratio(a, b) == 0.0
        assert not has_changed(a, b)

    def test_moved_piece_is_a_change(self) -> None:
        a = board_signature(make_frame())
        b = board_signature(make_frame(block=(50, 50, 100, 100)))
        assert changed_ratio(a, b) >= DEFAULT_CHANGE_RATIO
        assert has_changed(a, b)

    def test_single_cursor_pixel_is_not_a_change(self) -> None:
        a = board_signature(make_frame())
        b = board_signature(make_frame(noise=3))
        assert not has_changed(a, b)

    def test_no_baseline_reports_no_change(self) -> None:
        current = board_signature(make_frame())
        assert not has_changed(current, None)
        assert not has_changed(None, current)
        assert changed_ratio(current, None) is None

    def test_shape_mismatch_reports_change(self) -> None:
        current = board_signature(make_frame())
        previous = np.zeros((50, 50), dtype=np.uint8)
        assert has_changed(current, previous)

    def test_threshold_is_configurable(self) -> None:
        a = board_signature(make_frame())
        b = board_signature(make_frame(block=(50, 50, 100, 100)))
        assert not has_changed(a, b, ratio_threshold=1.0)


class TestWatchConfig:
    def test_defaults(self) -> None:
        config = WatchConfig()
        assert config.interval_ms == DEFAULT_INTERVAL_MS == 1000
        assert config.change_ratio == DEFAULT_CHANGE_RATIO

    def test_interval_is_configurable(self) -> None:
        assert WatchConfig(interval_ms=250).interval_ms == 250

    def test_rejects_aggressive_interval(self) -> None:
        with pytest.raises(ValueError):
            WatchConfig(interval_ms=MIN_INTERVAL_MS - 1)

    def test_rejects_zero_interval(self) -> None:
        with pytest.raises(ValueError):
            WatchConfig(interval_ms=0)

    def test_rejects_out_of_range_ratio(self) -> None:
        with pytest.raises(ValueError):
            WatchConfig(change_ratio=0.0)
        with pytest.raises(ValueError):
            WatchConfig(change_ratio=1.5)

    def test_confidence_threshold_matches_the_spec_default(self) -> None:
        assert WatchConfig().min_square_confidence == 0.60

    def test_confidence_threshold_is_configurable(self) -> None:
        assert WatchConfig(min_square_confidence=0.0).min_square_confidence == 0.0

    @pytest.mark.parametrize("value", [-0.1, 1.5])
    def test_rejects_out_of_range_confidence(self, value: float) -> None:
        with pytest.raises(ValueError):
            WatchConfig(min_square_confidence=value)


def build_watcher(
    frame: np.ndarray,
    recognizer: FakeRecognizer,
    captured: list[np.ndarray],
    config: WatchConfig | None = None,
) -> BoardWatcher:
    """Watcher over a region whose corners already cover the whole frame."""
    region = ScreenRegion(0, 0, REGION_SIZE, REGION_SIZE)
    watcher = BoardWatcher(
        region=region,
        board=make_board(frame),
        recognizer=recognizer,  # type: ignore[arg-type]
        config=config or WatchConfig(interval_ms=MIN_INTERVAL_MS),
    )
    watcher._capture_fn = lambda _region: captured[0]
    return watcher


class TestBoardWatcher:
    def _make_watcher(
        self,
        qapp,
        frame: np.ndarray,
        recognizer: FakeRecognizer,
        captured: list[np.ndarray],
    ) -> BoardWatcher:
        return build_watcher(frame, recognizer, captured)

    def test_requires_a_detected_board(self, qapp) -> None:
        region = ScreenRegion(0, 0, REGION_SIZE, REGION_SIZE)
        with pytest.raises(ValueError):
            BoardWatcher(region, None, FakeRecognizer())  # type: ignore[arg-type]

    def test_unchanged_frame_is_discarded(self, qapp) -> None:
        recognizer = FakeRecognizer()
        captured = [make_frame()]
        watcher = self._make_watcher(qapp, make_frame(), recognizer, captured)
        received: list[tuple[list, str]] = []
        watcher.position_ready.connect(lambda squares, fen: received.append((squares, fen)))

        assert watcher.poll_once() is False
        assert received == []
        assert recognizer.calls == 0

    def test_changed_frame_is_recognized(self, qapp) -> None:
        recognizer = FakeRecognizer()
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = self._make_watcher(qapp, make_frame(), recognizer, captured)
        received: list[tuple[list, str]] = []
        watcher.position_ready.connect(lambda squares, fen: received.append((squares, fen)))

        assert watcher.poll_once() is True
        assert recognizer.calls == 1
        assert len(received) == 1
        squares, fen = received[0]
        assert len(squares) == 64
        assert positions_equal(fen, START_FEN)

    def test_same_position_is_only_reported_once(self, qapp) -> None:
        recognizer = FakeRecognizer()
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = self._make_watcher(qapp, make_frame(), recognizer, captured)
        received: list[tuple[list, str]] = []
        watcher.position_ready.connect(lambda squares, fen: received.append((squares, fen)))

        assert watcher.poll_once() is True
        assert watcher.poll_once() is False
        assert recognizer.calls == 1
        assert len(received) == 1

    def test_capture_failure_is_reported_not_raised(self, qapp) -> None:
        recognizer = FakeRecognizer()
        captured: list[np.ndarray] = []
        watcher = self._make_watcher(qapp, make_frame(), recognizer, captured)
        failures: list[str] = []
        watcher.failure.connect(failures.append)
        watcher._capture_fn = lambda _region: (_ for _ in ()).throw(OSError("no screen"))

        assert watcher.poll_once() is False
        assert failures and "no screen" in failures[0]
        assert recognizer.calls == 0

    def test_recognition_failure_keeps_baseline_for_retry(self, qapp) -> None:
        recognizer = FakeRecognizer(error=PieceRecognizerError("model missing"))
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = self._make_watcher(qapp, make_frame(), recognizer, captured)
        failures: list[str] = []
        watcher.failure.connect(failures.append)

        assert watcher.poll_once() is False
        assert failures and "model missing" in failures[0]
        assert recognizer.calls == 1

        recognizer.error = None
        assert watcher.poll_once() is True, "failed frame must be retried next tick"
        assert recognizer.calls == 2

    def test_never_reruns_board_detection(self, qapp, monkeypatch) -> None:
        recognizer = FakeRecognizer()
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = self._make_watcher(qapp, make_frame(), recognizer, captured)

        def boom(_image):
            raise AssertionError("board detection must not run during watch")

        monkeypatch.setattr("app.vision.board_detector.detect_board", boom)

        assert watcher.poll_once() is True
        assert recognizer.calls == 1

    def test_stop_is_safe_when_never_started(self, qapp) -> None:
        recognizer = FakeRecognizer()
        captured = [make_frame()]
        watcher = self._make_watcher(qapp, make_frame(), recognizer, captured)
        watcher.stop()
        assert watcher.stop_requested is True


class TestWatchValidation:
    """Phase 7 inside the watch loop: garbage never reaches Stockfish."""

    @staticmethod
    def _collect(watcher: BoardWatcher):
        positions: list[tuple[list, str]] = []
        failures: list[str] = []
        watcher.position_ready.connect(lambda s, f: positions.append((s, f)))
        watcher.failure.connect(failures.append)
        return positions, failures

    def test_impossible_position_is_reported_not_emitted(self, qapp) -> None:
        recognizer = FakeRecognizer(squares=fen_to_squares("8/8/8/8/8/8/8/8"))
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = build_watcher(make_frame(), recognizer, captured)
        positions, failures = self._collect(watcher)

        assert watcher.poll_once() is False
        assert positions == []
        assert failures[0].split("\n")[0] == INVALID_POSITION_MESSAGE
        assert recognizer.calls == 1

    def test_rejected_frame_is_not_recognized_again(self, qapp) -> None:
        recognizer = FakeRecognizer(squares=fen_to_squares("8/8/8/8/8/8/8/8"))
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = build_watcher(make_frame(), recognizer, captured)
        self._collect(watcher)

        assert watcher.poll_once() is False
        assert watcher.poll_once() is False
        assert recognizer.calls == 1, "the same frame must not be re-examined"

    def test_frame_returning_to_the_known_position_is_a_temporary_change(
        self, qapp
    ) -> None:
        recognizer = FakeRecognizer(squares=fen_to_squares("8/8/8/8/8/8/8/8"))
        captured: list[np.ndarray] = [make_frame(block=(50, 50, 100, 100))]
        watcher = build_watcher(make_frame(), recognizer, captured)
        _, failures = self._collect(watcher)

        assert watcher.poll_once() is False
        assert failures[-1].split("\n")[0] == INVALID_POSITION_MESSAGE

        captured[0] = make_frame()  # the pixels came back to the baseline
        assert watcher.poll_once() is False
        assert failures[-1] == TEMPORARY_VISUAL_MESSAGE
        assert recognizer.calls == 1

    def test_low_confidence_blocks_the_position(self, qapp) -> None:
        confidences = [0.0] * 64
        confidences[60] = 0.30  # K on e1
        confidences[63] = 0.90  # k on h1
        recognizer = FakeRecognizer(confidences=confidences)
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = build_watcher(make_frame(), recognizer, captured)
        positions, failures = self._collect(watcher)

        assert watcher.poll_once() is False
        assert positions == []
        assert failures[0].startswith(LOW_CONFIDENCE_HEADER)
        assert "e1 — white king — 0.30" in failures[0]

    def test_confidence_check_can_be_switched_off(self, qapp) -> None:
        confidences = [0.0] * 64
        confidences[60] = 0.05
        confidences[63] = 0.05
        recognizer = FakeRecognizer(confidences=confidences)
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = build_watcher(
            make_frame(),
            recognizer,
            captured,
            config=WatchConfig(
                interval_ms=MIN_INTERVAL_MS, min_square_confidence=0.0
            ),
        )
        positions, failures = self._collect(watcher)

        assert watcher.poll_once() is True
        assert failures == []
        assert len(positions) == 1

    def test_same_failure_is_not_repeated_on_every_tick(self, qapp) -> None:
        recognizer = FakeRecognizer()
        captured: list[np.ndarray] = []
        watcher = build_watcher(make_frame(), recognizer, captured)
        watcher._capture_fn = lambda _region: (_ for _ in ()).throw(OSError("no screen"))
        _, failures = self._collect(watcher)

        assert watcher.poll_once() is False
        assert watcher.poll_once() is False
        assert len(failures) == 1

    def test_a_later_different_failure_is_still_reported(self, qapp) -> None:
        recognizer = FakeRecognizer(error=PieceRecognizerError("model missing"))
        captured = [make_frame(block=(50, 50, 100, 100))]
        watcher = build_watcher(make_frame(), recognizer, captured)
        watcher._capture_fn = lambda _region: (_ for _ in ()).throw(OSError("no screen"))
        _, failures = self._collect(watcher)

        watcher.poll_once()
        watcher._capture_fn = lambda _region: captured[0]
        watcher.poll_once()
        watcher._capture_fn = lambda _region: (_ for _ in ()).throw(OSError("no screen"))
        watcher.poll_once()

        assert len(failures) == 3
        assert "no screen" in failures[0]
        assert "model missing" in failures[1]
        assert "no screen" in failures[2]
