from pathlib import Path

import chess
import cv2
import numpy as np
import pytest

from app.vision.piece_recognizer import (
    BOARD_CLASS,
    DEFAULT_WEIGHTS_PATH,
    EMPTY_SQUARE,
    PIECE_SYMBOLS,
    PieceRecognizer,
    PieceRecognizerError,
    assign_to_squares,
    format_piece_map,
    symbol_from_class_name,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "board_diagram.png"
FIXTURE_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


def expected_squares(fen: str) -> list[str]:
    board = chess.Board(fen)
    squares = []
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            squares.append("." if piece is None else piece.symbol())
    return squares


class TestSymbolFromClassName:
    def test_all_twelve_pieces_mapped(self) -> None:
        assert len(PIECE_SYMBOLS) == 12
        for name, symbol in PIECE_SYMBOLS.items():
            assert symbol_from_class_name(name) == symbol

    def test_white_pieces_are_uppercase(self) -> None:
        for name, symbol in PIECE_SYMBOLS.items():
            if name.startswith("white_"):
                assert symbol.isupper()

    def test_black_pieces_are_lowercase(self) -> None:
        for name, symbol in PIECE_SYMBOLS.items():
            if name.startswith("black_"):
                assert symbol.islower()

    def test_piece_types_use_standard_fen_letters(self) -> None:
        letters = set(PIECE_SYMBOLS.values())
        assert letters == {"K", "Q", "R", "B", "N", "P", "k", "q", "r", "b", "n", "p"}

    def test_board_class_returns_none(self) -> None:
        assert symbol_from_class_name(BOARD_CLASS) is None

    def test_unknown_class_returns_none(self) -> None:
        assert symbol_from_class_name("something_else") is None


class TestAssignToSquares:
    def test_returns_64_empty_without_detections(self) -> None:
        squares = assign_to_squares([], width=640, height=640)
        assert len(squares) == 64
        assert squares == [EMPTY_SQUARE] * 64

    def test_center_detection_lands_on_e4(self) -> None:
        # index 36 == e4
        squares = assign_to_squares(
            [("white_pawn", 0.9, (300, 300, 340, 340))], width=640, height=640
        )
        assert squares[36] == "P"

    def test_corners_map_to_a8_and_h1(self) -> None:
        squares = assign_to_squares(
            [
                ("white_rook", 0.9, (0, 0, 20, 20)),          # top-left  -> a8
                ("black_rook", 0.9, (620, 620, 640, 640)),    # bottom-right -> h1
            ],
            width=640,
            height=640,
        )
        assert squares[0] == "R"
        assert squares[63] == "r"

    def test_squares_follow_a8_to_h1_order(self) -> None:
        squares = assign_to_squares(
            [
                ("white_pawn", 0.9, (10, 10, 20, 20)),    # a8 -> index 0
                ("white_pawn", 0.9, (630, 10, 640, 20)),  # h8 -> index 7
                ("white_pawn", 0.9, (10, 630, 20, 640)),  # a1 -> index 56
                ("white_pawn", 0.9, (630, 630, 640, 640)),  # h1 -> index 63
            ],
            width=640,
            height=640,
        )
        assert [squares[i] for i in (0, 7, 56, 63)] == ["P", "P", "P", "P"]

    def test_board_class_is_ignored(self) -> None:
        squares = assign_to_squares(
            [(BOARD_CLASS, 0.9, (0, 0, 640, 640))], width=640, height=640
        )
        assert squares == [EMPTY_SQUARE] * 64

    def test_unknown_class_is_ignored(self) -> None:
        squares = assign_to_squares(
            [("mystery", 0.9, (300, 300, 340, 340))], width=640, height=640
        )
        assert squares == [EMPTY_SQUARE] * 64

    def test_out_of_bounds_detection_is_ignored(self) -> None:
        squares = assign_to_squares(
            [("white_pawn", 0.9, (640, 640, 660, 660))], width=640, height=640
        )
        assert squares == [EMPTY_SQUARE] * 64

    def test_highest_confidence_wins_on_conflict(self) -> None:
        box = (300, 300, 340, 340)
        squares = assign_to_squares(
            [
                ("white_knight", 0.4, box),
                ("white_pawn", 0.9, box),
            ],
            width=640,
            height=640,
        )
        assert squares[36] == "P"

    def test_lower_confidence_does_not_overwrite(self) -> None:
        box = (300, 300, 340, 340)
        squares = assign_to_squares(
            [
                ("white_pawn", 0.9, box),
                ("black_knight", 0.2, box),
            ],
            width=640,
            height=640,
        )
        assert squares[36] == "P"

    def test_rejects_non_positive_size(self) -> None:
        with pytest.raises(ValueError):
            assign_to_squares([], width=0, height=640)

    def test_handles_board_image_at_native_size(self) -> None:
        # center (450, 450) on an 800x800 board -> row 4, col 4 -> index 36 (e4)
        squares = assign_to_squares(
            [("black_queen", 0.9, (400, 400, 500, 500))], width=800, height=800
        )
        assert squares[36] == "q"


class TestFormatPieceMap:
    def test_renders_eight_rows_of_eight(self) -> None:
        lines = format_piece_map(["."] * 64).split("\n")
        assert len(lines) == 8
        assert all(len(line.split(" ")) == 8 for line in lines)

    def test_places_symbols_by_index(self) -> None:
        squares = ["."] * 64
        squares[0] = "R"
        squares[63] = "r"
        lines = format_piece_map(squares).split("\n")
        assert lines[0].split(" ")[0] == "R"
        assert lines[7].split(" ")[7] == "r"


class TestPieceRecognizer:
    def test_empty_image_raises(self) -> None:
        recognizer = PieceRecognizer(weights_path="/nonexistent/model.pt")
        with pytest.raises(PieceRecognizerError):
            recognizer.recognize(np.array([]))

    def test_none_image_raises(self) -> None:
        recognizer = PieceRecognizer(weights_path="/nonexistent/model.pt")
        with pytest.raises(PieceRecognizerError):
            recognizer.recognize(None)

    def test_missing_weights_raises_helpful_error(self) -> None:
        recognizer = PieceRecognizer(weights_path="/nonexistent/model.pt")
        assert not recognizer.is_loaded
        with pytest.raises(PieceRecognizerError) as exc:
            recognizer.load()
        assert "weights not found" in str(exc.value)
        assert "huggingface.co" in str(exc.value)

    def test_recognize_board_without_board_raises(self) -> None:
        recognizer = PieceRecognizer(weights_path="/nonexistent/model.pt")
        with pytest.raises(PieceRecognizerError):
            recognizer.recognize_board(None)

    def test_defaults_point_to_project_models_directory(self) -> None:
        recognizer = PieceRecognizer()
        assert recognizer.weights_path == DEFAULT_WEIGHTS_PATH
        assert recognizer.weights_path.name == "nakst-chess-2d-best.pt"
        assert recognizer.confidence == 0.25
        assert not recognizer.is_loaded


@pytest.mark.skipif(
    not DEFAULT_WEIGHTS_PATH.exists(),
    reason="model weights not downloaded",
)
class TestRecognizeIntegration:
    def test_recognizes_fixture_board(self) -> None:
        image = cv2.imread(str(FIXTURE_PATH))
        assert image is not None
        assert image.shape[:2] == (800, 800)

        recognizer = PieceRecognizer()
        squares = recognizer.recognize(image)

        assert len(squares) == 64
        truth = expected_squares(FIXTURE_FEN)
        correct = sum(1 for got, want in zip(squares, truth) if got == want)
        assert correct / 64 >= 0.90, f"only {correct}/64 squares correct: {squares}"

    def test_recognize_board_uses_transformed_image(self) -> None:
        from app.vision.board_detector import detect_board

        image = cv2.imread(str(FIXTURE_PATH))
        board = detect_board(image)
        assert board is not None

        recognizer = PieceRecognizer()
        squares = recognizer.recognize_board(board)
        assert len(squares) == 64
        truth = expected_squares(FIXTURE_FEN)
        correct = sum(1 for got, want in zip(squares, truth) if got == want)
        assert correct / 64 >= 0.90
