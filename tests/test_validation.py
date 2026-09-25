"""Phase 7: nothing reaches Stockfish unless the position is valid."""

from __future__ import annotations

import chess
import pytest

from app.chess.validation import (
    BOARD_DETECTION_MESSAGE,
    DEFAULT_MIN_SQUARE_CONFIDENCE,
    INVALID_POSITION_MESSAGE,
    TEMPORARY_VISUAL_MESSAGE,
    ErrorKind,
    ValidationResult,
    board_detection_failure,
    temporary_visual_change,
    validate_fen,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MID_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
KINGS_ONLY = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


class TestValidPositions:
    @pytest.mark.parametrize("fen", [START_FEN, MID_FEN, KINGS_ONLY])
    def test_known_good_fens_pass(self, fen: str) -> None:
        result = validate_fen(fen)
        assert result.ok
        assert result.is_valid
        assert bool(result)
        assert result.kind is None
        assert result.display == ""

    def test_result_carries_a_parseable_board(self) -> None:
        result = validate_fen(START_FEN)
        assert isinstance(result.board, chess.Board)
        assert result.board.king(chess.WHITE) == chess.E1
        assert result.fen == START_FEN


class TestInvalidFen:
    @pytest.mark.parametrize(
        "fen",
        [
            "not a fen",
            "",
            "   ",
            "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
            "4k3/9/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",
        ],
    )
    def test_unparseable_fen_is_rejected(self, fen: str) -> None:
        result = validate_fen(fen)
        assert not result.ok
        assert result.kind is ErrorKind.INVALID_FEN
        assert result.board is None

    def test_message_is_the_text_from_the_spec(self) -> None:
        assert validate_fen("garbage").message == INVALID_POSITION_MESSAGE
        assert validate_fen("").message == INVALID_POSITION_MESSAGE

    def test_detail_says_what_went_wrong(self) -> None:
        result = validate_fen("not a fen")
        assert "turn" in result.detail

    def test_display_shows_message_then_detail(self) -> None:
        display = validate_fen("not a fen").display
        assert display.split("\n")[0] == INVALID_POSITION_MESSAGE
        assert "turn" in display


class TestExactlyOneKingPerSide:
    def test_missing_white_king(self) -> None:
        result = validate_fen("4k3/8/8/8/8/8/8/8 w - - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "white king" in result.detail

    def test_missing_black_king(self) -> None:
        result = validate_fen("8/8/8/8/8/8/8/4K3 w - - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "black king" in result.detail

    def test_two_white_kings(self) -> None:
        result = validate_fen("4k3/8/8/8/8/8/8/K6K w - - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "exactly one king per side" in result.detail

    def test_empty_board_lists_every_problem(self) -> None:
        result = validate_fen("8/8/8/8/8/8/8/8 w - - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "white king" in result.detail and "black king" in result.detail

    @pytest.mark.parametrize("side", ["w", "b"])
    def test_position_valid_for_either_side_to_move(self, side: str) -> None:
        fen = f"4k3/8/8/8/8/8/8/4K3 {side} - - 0 1"
        assert validate_fen(fen, side_to_move=side).ok


class TestSideToMove:
    def test_agreeing_side_to_move_is_accepted(self) -> None:
        assert validate_fen(START_FEN, side_to_move="w").ok
        assert validate_fen(MID_FEN, side_to_move="b").ok

    @pytest.mark.parametrize("side", ["x", "", "W", "both"])
    def test_malformed_side_to_move_is_invalid(self, side: str) -> None:
        result = validate_fen(START_FEN, side_to_move=side)
        assert result.kind is ErrorKind.INVALID_FEN

    def test_side_to_move_contradicting_the_fen_is_invalid(self) -> None:
        result = validate_fen(START_FEN, side_to_move="b")
        assert result.kind is ErrorKind.INVALID_FEN
        assert "contradicts" in result.detail

    def test_side_not_to_move_already_in_check(self) -> None:
        # Black is to move but white's rook already checks the black king.
        result = validate_fen("4k3/8/8/8/8/8/8/r3K3 b - - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "side not to move" in result.detail


class TestObviouslyImpossible:
    def test_pawn_on_the_back_rank(self) -> None:
        result = validate_fen("4k3/8/8/8/8/8/8/P3K3 w - - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "rank 1 or 8" in result.detail

    def test_nine_pawns(self) -> None:
        result = validate_fen("4k3/8/8/8/PPPPPPPP/PPPPPPPP/4K3/8 w - - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "more than 8 white pawns" in result.detail

    def test_impossible_castling_rights(self) -> None:
        # White claims kingside castling but has no rook on h1.
        result = validate_fen("4k3/8/8/8/8/8/8/4K3 w K - 0 1")
        assert result.kind is ErrorKind.INVALID_POSITION
        assert "castling" in result.detail


class TestLowConfidence:
    @staticmethod
    def _squares_with_kings() -> list[str]:
        squares = ["."] * 64
        squares[60] = "K"  # e1
        squares[63] = "k"  # h1
        return squares

    def test_doubtful_detection_blocks_the_position(self) -> None:
        confidences = [0.0] * 64
        confidences[60] = 0.54
        confidences[63] = 0.93
        result = validate_fen(
            KINGS_ONLY,
            squares=self._squares_with_kings(),
            confidences=confidences,
        )
        assert result.kind is ErrorKind.LOW_CONFIDENCE
        assert not result.ok

    def test_detail_matches_the_format_from_the_spec(self) -> None:
        squares = ["."] * 64
        squares[36] = "b"  # e4
        confidences = [0.0] * 64
        confidences[36] = 0.54
        result = validate_fen(
            "4k3/8/8/8/4b3/8/8/4K3 w - - 0 1",
            squares=squares,
            confidences=confidences,
        )
        assert result.kind is ErrorKind.LOW_CONFIDENCE
        assert result.detail == "e4 — black bishop — 0.54"
        assert result.message.startswith("Position uncertain")
        assert "Please capture again." in result.display

    def test_confident_detections_are_accepted(self) -> None:
        confidences = [0.0] * 64
        confidences[60] = 0.93
        confidences[63] = 0.88
        result = validate_fen(
            KINGS_ONLY,
            squares=self._squares_with_kings(),
            confidences=confidences,
        )
        assert result.ok

    def test_empty_squares_never_count_as_low_confidence(self) -> None:
        result = validate_fen(
            KINGS_ONLY,
            squares=self._squares_with_kings(),
            confidences=[0.0] * 64,
            min_square_confidence=0.0,
        )
        assert result.ok

    def test_threshold_is_configurable_and_can_be_disabled(self) -> None:
        confidences = [0.0] * 64
        confidences[60] = 0.05
        kwargs = {
            "squares": self._squares_with_kings(),
            "confidences": confidences,
        }
        assert validate_fen(KINGS_ONLY, min_square_confidence=0.0, **kwargs).ok
        assert not validate_fen(
            KINGS_ONLY, min_square_confidence=0.99, **kwargs
        ).ok

    def test_structural_problems_are_reported_before_confidence(self) -> None:
        result = validate_fen(
            "8/8/8/8/8/8/8/8 w - - 0 1",
            squares=["."] * 64,
            confidences=[0.0] * 64,
        )
        assert result.kind is ErrorKind.INVALID_POSITION

    def test_default_threshold_is_the_configured_one(self) -> None:
        assert DEFAULT_MIN_SQUARE_CONFIDENCE == 0.60


class TestOtherCategories:
    def test_board_detection_failure(self) -> None:
        result = board_detection_failure()
        assert result.kind is ErrorKind.BOARD_DETECTION_FAILURE
        assert result.display == BOARD_DETECTION_MESSAGE
        assert not result.ok

    def test_temporary_visual_change(self) -> None:
        result = temporary_visual_change()
        assert result.kind is ErrorKind.TEMPORARY_VISUAL_CHANGE
        assert result.display == TEMPORARY_VISUAL_MESSAGE
        assert not result.ok

    def test_kinds_carry_the_distinguishing_value(self) -> None:
        assert ErrorKind.INVALID_FEN.value == "invalid_fen"
        assert ErrorKind.LOW_CONFIDENCE.value == "low_confidence"
        assert ErrorKind.BOARD_DETECTION_FAILURE.value == "board_detection_failure"
        assert ErrorKind.TEMPORARY_VISUAL_CHANGE.value == "temporary_visual_change"


class TestSquareNaming:
    @pytest.mark.parametrize(
        "index, expected",
        [(0, "a8"), (7, "h8"), (36, "e4"), (60, "e1"), (63, "h1")],
    )
    def test_index_maps_to_the_right_square(self, index: int, expected: str) -> None:
        from app.chess.validation import _describe_square

        assert _describe_square(index, "K", 0.9).split(" — ")[0] == expected

    @pytest.mark.parametrize(
        "symbol, expected",
        [("K", "white king"), ("b", "black bishop"), ("p", "black pawn"), ("Q", "white queen")],
    )
    def test_piece_is_named_for_humans(self, symbol: str, expected: str) -> None:
        from app.chess.validation import _describe_square

        assert expected in _describe_square(36, symbol, 0.54)


def test_result_is_false_when_invalid() -> None:
    assert not validate_fen("garbage")
    assert ValidationResult(ok=True)


def test_invalid_result_has_no_display_when_valid() -> None:
    assert ValidationResult(ok=True).display == ""
