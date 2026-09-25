"""Watch loop: periodic capture, cheap change gate, recognition on change.

Implements the Phase 6/7 flow::

    capture -> cheap image comparison -> did the board change?
        NO  -> do nothing
        YES -> recognize -> FEN -> validate -> Stockfish

Full board detection is never rerun: the corners stored by Phase 4 are reused
for every frame, and YOLO only runs on frames whose signature differs from the
one that produced the current position. Validation (Phase 7) happens *before*
any position is reported, so an impossible board never reaches the engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from PySide6.QtCore import QThread, Signal

from app.capture.screen_capture import ScreenRegion, capture_region
from app.chess.position import piece_map_to_fen
from app.chess.validation import (
    DEFAULT_MIN_SQUARE_CONFIDENCE,
    board_detection_failure,
    temporary_visual_change,
    validate_fen,
)
from app.vision.board_detector import DetectedBoard
from app.vision.change_detector import DEFAULT_CHANGE_RATIO, board_signature, has_changed
from app.vision.piece_recognizer import PieceRecognizer, PieceRecognizerError

# 1 s is slow enough to keep CPU idle on an i7-1065G7 and fast enough that a
# move is noticed while the user is still looking at the board.
DEFAULT_INTERVAL_MS = 1000
MIN_INTERVAL_MS = 100


@dataclass(frozen=True)
class WatchConfig:
    """Monitoring parameters. Every value is configurable per the spec."""

    interval_ms: int = DEFAULT_INTERVAL_MS
    change_ratio: float = DEFAULT_CHANGE_RATIO
    # Phase 7: a square recognized with less confidence than this marks the
    # whole position uncertain instead of letting it reach Stockfish. 0 turns
    # the check off.
    min_square_confidence: float = DEFAULT_MIN_SQUARE_CONFIDENCE

    def __post_init__(self) -> None:
        if self.interval_ms < MIN_INTERVAL_MS:
            raise ValueError(
                f"interval_ms must be at least {MIN_INTERVAL_MS}, "
                f"got {self.interval_ms}"
            )
        if not 0.0 < self.change_ratio <= 1.0:
            raise ValueError(f"change_ratio must be in (0, 1], got {self.change_ratio}")
        if not 0.0 <= self.min_square_confidence <= 1.0:
            raise ValueError(
                "min_square_confidence must be in [0, 1], "
                f"got {self.min_square_confidence}"
            )


class BoardWatcher(QThread):
    """Polls the selected region and reports a new position when the board changes."""

    position_ready = Signal(list, str)
    failure = Signal(str)

    def __init__(
        self,
        region: ScreenRegion,
        board: DetectedBoard,
        recognizer: PieceRecognizer,
        config: WatchConfig | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        if board is None:
            raise ValueError("A detected board is required to watch.")
        self._region = region
        self._board = board
        self._recognizer = recognizer
        self._config = config or WatchConfig()
        self._stop_requested = False

        baseline = board_signature(board.transformed)
        # Signature of the last position that passed validation. A frame that
        # comes back to it after a rejection was a temporary visual change.
        self._known = baseline
        # Signature of the last frame recognition was run on. Stops YOLO from
        # being rerun on an image that has already been examined (and, if it
        # was rejected, would be rejected again).
        self._attempted = baseline
        self._invalid_pending = False
        self._last_failure: str | None = None

        # Indirection so tests can substitute a deterministic "screen".
        self._capture_fn = capture_region

    @property
    def config(self) -> WatchConfig:
        return self._config

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        while not self._stop_requested:
            self.poll_once()
            self._sleep_until_next_tick()

    def poll_once(self) -> bool:
        """Capture, gate, recognize, validate. True when a position was emitted."""
        try:
            frame = self._capture_fn(self._region)
        except Exception as exc:  # capture must never kill the loop
            self._emit_failure(f"Capture failed:\n{exc}")
            return False

        normalized = self._board.transform_capture(frame)
        if normalized is None:
            self._emit_failure(board_detection_failure().display)
            return False

        signature = board_signature(normalized)

        if self._attempted is not None and not has_changed(
            signature, self._attempted, self._config.change_ratio
        ):
            return False  # already examined this exact frame

        if self._known is not None and not has_changed(
            signature, self._known, self._config.change_ratio
        ):
            # The pixels came back to the last accepted position: whatever made
            # them differ a moment ago (cursor, highlight, animation) is gone.
            self._attempted = signature
            if self._invalid_pending:
                self._invalid_pending = False
                self._emit_failure(temporary_visual_change().display)
            return False

        self._attempted = signature
        try:
            squares, confidences = self._recognizer.recognize_detailed(normalized)
        except PieceRecognizerError as exc:
            # Forget the frame so the same image is retried next tick, but
            # keep the last valid baseline untouched.
            self._attempted = None
            self._emit_failure(str(exc))
            return False

        fen = piece_map_to_fen(squares)
        result = validate_fen(
            fen,
            squares=squares,
            confidences=confidences,
            min_square_confidence=self._config.min_square_confidence,
        )
        if not result.ok:
            self._invalid_pending = True
            self._emit_failure(result.display)
            return False

        self._known = signature
        self._invalid_pending = False
        self._last_failure = None
        self.position_ready.emit(squares, fen)
        return True

    def _emit_failure(self, message: str) -> None:
        """Report a failure once, not on every tick until it goes away."""
        if message == self._last_failure:
            return
        self._last_failure = message
        self.failure.emit(message)

    def _sleep_until_next_tick(self) -> None:
        remaining = self._config.interval_ms / 1000.0
        while remaining > 0 and not self._stop_requested:
            step = min(0.05, remaining)
            time.sleep(step)
            remaining -= step
