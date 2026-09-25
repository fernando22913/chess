from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from app.vision.board_detector import DetectedBoard

MODEL_INPUT_SIZE = 640
DEFAULT_CONFIDENCE = 0.25
EMPTY_SQUARE = "."
BOARD_CLASS = "board"

# Model class name -> FEN piece symbol (uppercase = white, lowercase = black).
PIECE_SYMBOLS: dict[str, str] = {
    "white_king": "K",
    "white_queen": "Q",
    "white_rook": "R",
    "white_bishop": "B",
    "white_knight": "N",
    "white_pawn": "P",
    "black_king": "k",
    "black_queen": "q",
    "black_rook": "r",
    "black_bishop": "b",
    "black_knight": "n",
    "black_pawn": "p",
}

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "nakst-chess-2d-best.pt"
)


class PieceRecognizerError(RuntimeError):
    """Raised when the piece recognizer cannot be loaded or cannot run."""


def symbol_from_class_name(name: str) -> str | None:
    """Map a model class name to its FEN symbol.

    Returns ``None`` for the ``board`` class and for unknown names.
    """
    return PIECE_SYMBOLS.get(name)


def assign_to_squares_detailed(
    detections: Iterable[tuple[str, float, Sequence[float]]],
    width: int,
    height: int,
) -> tuple[list[str], list[float]]:
    """Map detections to 64 squares and return their confidences too.

    Same rules as :func:`assign_to_squares`; the second list holds the winning
    confidence per square and ``0.0`` for squares with no piece, so the caller
    can tell a sure detection from a doubtful one.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    squares = [EMPTY_SQUARE] * 64
    confidences = [0.0] * 64

    for class_name, confidence, box in detections:
        symbol = symbol_from_class_name(class_name)
        if symbol is None:
            continue

        x1, y1, x2, y2 = box
        center_x = (float(x1) + float(x2)) / 2.0
        center_y = (float(y1) + float(y2)) / 2.0

        col = int(center_x / width * 8)
        row = int(center_y / height * 8)
        if not (0 <= row < 8 and 0 <= col < 8):
            continue

        index = row * 8 + col
        if confidence > confidences[index]:
            squares[index] = symbol
            confidences[index] = float(confidence)

    return squares, confidences


def assign_to_squares(
    detections: Iterable[tuple[str, float, Sequence[float]]],
    width: int,
    height: int,
) -> list[str]:
    """Map detections to 64 squares ordered a8..h1.

    Each detection is ``(class_name, confidence, (x1, y1, x2, y2))`` in pixel
    coordinates of an image of size ``width`` x ``height`` that contains only
    the board. Squares with no detection stay empty; when several detections
    land on the same square, the highest confidence wins.
    """
    squares, _ = assign_to_squares_detailed(detections, width, height)
    return squares


def format_piece_map(squares: Sequence[str]) -> str:
    """Render 64 squares (a8..h1) as eight rows of eight symbols."""
    return "\n".join(
        " ".join(squares[row * 8 + col] for col in range(8)) for row in range(8)
    )


class PieceRecognizer:
    """Recognize the pieces on a normalized board using a YOLO detector.

    The model is loaded lazily on the first :meth:`recognize` call, so
    constructing a recognizer never imports torch.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        self._weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH
        self._confidence = confidence
        self._model = None

    @property
    def weights_path(self) -> Path:
        return self._weights_path

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def confidence(self) -> float:
        return self._confidence

    def load(self) -> None:
        """Load the YOLO model. Idempotent."""
        if self._model is not None:
            return
        if not self._weights_path.exists():
            raise PieceRecognizerError(
                "Model weights not found: "
                f"{self._weights_path}\n"
                "Download them into the project 'models' directory:\n"
                "  curl -L -o models/nakst-chess-2d-best.pt "
                "https://huggingface.co/NAKSTStudio/yolov8m-chess-piece-detection/"
                "resolve/main/best.pt"
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise PieceRecognizerError(
                "ultralytics is not installed; piece recognition is unavailable."
            ) from exc
        self._model = YOLO(str(self._weights_path))

    def recognize(self, image: np.ndarray) -> list[str]:
        """Recognize pieces on a board image and return 64 symbols (a8..h1).

        ``image`` must be a board-filling image (``DetectedBoard.transformed``
        is the expected input). Returns one symbol per square; squares without
        a piece are ``"."``.
        """
        squares, _ = self.recognize_detailed(image)
        return squares

    def recognize_detailed(
        self, image: np.ndarray
    ) -> tuple[list[str], list[float]]:
        """Like :meth:`recognize`, also returning the confidence per square.

        The second element is 0.0 for empty squares. Phase 7 uses it to keep a
        doubtful detection from reaching Stockfish.
        """
        if image is None or getattr(image, "size", 0) == 0:
            raise PieceRecognizerError("Cannot recognize pieces on an empty image.")

        self.load()

        results = self._model.predict(
            image,
            imgsz=MODEL_INPUT_SIZE,
            conf=self._confidence,
            verbose=False,
        )[0]

        height, width = results.orig_shape
        detections: list[tuple[str, float, Sequence[float]]] = []
        for box in results.boxes:
            class_name = self._model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            xyxy = [float(value) for value in box.xyxy[0]]
            detections.append((class_name, confidence, xyxy))

        return assign_to_squares_detailed(detections, width=width, height=height)

    def recognize_board(self, board: DetectedBoard | None) -> list[str]:
        """Recognize pieces on a previously detected board."""
        squares, _ = self.recognize_board_detailed(board)
        return squares

    def recognize_board_detailed(
        self, board: DetectedBoard | None
    ) -> tuple[list[str], list[float]]:
        """Like :meth:`recognize_board`, also returning confidences."""
        if board is None:
            raise PieceRecognizerError("No detected board to recognize.")
        if board.transformed is None or board.transformed.size == 0:
            raise PieceRecognizerError("Detected board has no image data.")
        return self.recognize_detailed(board.transformed)
