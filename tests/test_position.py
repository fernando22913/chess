"""Tests for piece-map -> FEN conversion and position comparison."""

from __future__ import annotations

import chess
import pytest

from app.chess.position import (
    PositionError,
    castling_rights,
    normalize_fen,
    piece_map_to_fen,
    piece_map_to_placement,
    positions_equal,
)

ITALIAN_FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 3 3"


def squares_from_fen(fen: str) -> list[str]:
    """Convert a FEN into 64 symbols ordered a8..h1."""
    board = chess.Board(fen)
    squares = []
    for rank in range(7, -1, -1):
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            squares.append(piece.symbol() if piece else ".")
    return squares


def squares_of(*pairs: tuple[str, str]) -> list[str]:
    """Build a piece map from ``(square_name, symbol)`` pairs."""
    squares = ["."] * 64
    for name, symbol in pairs:
        file_index = ord(name[0]) - ord("a")
        rank_index = int(name[1])
        squares[(8 - rank_index) * 8 + file_index] = symbol
    return squares


class TestPlacement:
    def test_starting_position(self) -> None:
        squares = squares_from_fen(chess.STARTING_FEN)
        assert piece_map_to_placement(squares) == (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
        )

    def test_empty_board(self) -> None:
        assert piece_map_to_placement(["."] * 64) == "8/8/8/8/8/8/8/8"

    def test_single_piece(self) -> None:
        squares = squares_of(("e4", "P"))
        assert piece_map_to_placement(squares) == "8/8/8/8/4P3/8/8/8"

    def test_consecutive_empty_squares_are_numbered(self) -> None:
        squares = squares_of(("a8", "r"), ("h8", "r"))
        assert piece_map_to_placement(squares).split("/")[0] == "r6r"

    def test_rank_with_only_empties(self) -> None:
        squares = squares_of(("a1", "K"))
        assert piece_map_to_placement(squares).split("/")[7] == "K7"

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(PositionError):
            piece_map_to_placement(["."] * 63)

    def test_rejects_multicharacter_symbol(self) -> None:
        squares = ["."] * 64
        squares[0] = "PN"
        with pytest.raises(PositionError):
            piece_map_to_placement(squares)


class TestCastlingRights:
    def test_all_rights(self) -> None:
        squares = squares_from_fen(chess.STARTING_FEN)
        assert castling_rights(squares) == "KQkq"

    def test_no_rights(self) -> None:
        assert castling_rights(["."] * 64) == "-"

    def test_white_kingside_only(self) -> None:
        squares = squares_of(("e1", "K"), ("h1", "R"), ("e8", "k"))
        assert castling_rights(squares) == "K"

    def test_black_queenside_only(self) -> None:
        squares = squares_of(("e8", "k"), ("a8", "r"), ("e1", "K"))
        assert castling_rights(squares) == "q"

    def test_rook_on_wrong_square_grants_nothing(self) -> None:
        squares = squares_of(("e1", "K"), ("a2", "R"))
        assert castling_rights(squares) == "-"

    def test_canonical_order(self) -> None:
        squares = squares_of(
            ("e1", "K"), ("a1", "R"), ("h1", "R"),
            ("e8", "k"), ("a8", "r"), ("h8", "r"),
        )
        assert castling_rights(squares) == "KQkq"

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(PositionError):
            castling_rights(["."] * 10)


class TestFenGeneration:
    def test_starting_position_round_trip(self) -> None:
        squares = squares_from_fen(chess.STARTING_FEN)
        assert piece_map_to_fen(squares) == chess.STARTING_FEN

    def test_italian_game_round_trip(self) -> None:
        squares = squares_from_fen(ITALIAN_FEN)
        generated = piece_map_to_fen(squares)
        assert generated.split()[:4] == ITALIAN_FEN.split()[:4]

    def test_empty_board(self) -> None:
        assert piece_map_to_fen(["."] * 64) == "8/8/8/8/8/8/8/8 w - - 0 1"

    def test_black_to_move(self) -> None:
        squares = squares_from_fen(chess.STARTING_FEN)
        fen = piece_map_to_fen(squares, side_to_move="b")
        assert fen.split()[1] == "b"

    def test_rejects_unknown_side_to_move(self) -> None:
        with pytest.raises(PositionError):
            piece_map_to_fen(["."] * 64, side_to_move="x")

    def test_python_chess_accepts_generated_fen(self) -> None:
        squares = squares_from_fen(ITALIAN_FEN)
        board = chess.Board(piece_map_to_fen(squares))
        assert board.is_valid()


class TestPositionComparison:
    def test_normalize_drops_move_counters(self) -> None:
        a = "8/8/8/8/4P3/8/8/8 w - - 0 1"
        b = "8/8/8/8/4P3/8/8/8 w - - 17 42"
        assert normalize_fen(a) == normalize_fen(b)

    def test_same_position_different_counters_is_equal(self) -> None:
        a = "8/8/8/8/4P3/8/8/8 w - - 0 1"
        b = "8/8/8/8/4P3/8/8/8 w - - 55 90"
        assert positions_equal(a, b)

    def test_different_piece_is_not_equal(self) -> None:
        a = "8/8/8/8/4P3/8/8/8 w - - 0 1"
        b = "8/8/8/8/4N3/8/8/8 w - - 0 1"
        assert not positions_equal(a, b)

    def test_different_side_to_move_is_not_equal(self) -> None:
        a = "8/8/8/8/4P3/8/8/8 w - - 0 1"
        b = "8/8/8/8/4P3/8/8/8 b - - 0 1"
        assert not positions_equal(a, b)

    def test_different_placement_is_not_equal(self) -> None:
        a = "8/8/8/8/4P3/8/8/8 w - - 0 1"
        b = "8/8/8/8/3P4/8/8/8 w - - 0 1"
        assert not positions_equal(a, b)

    def test_rejects_truncated_fen(self) -> None:
        with pytest.raises(PositionError):
            normalize_fen("8/8/8/8/4P3/8/8/8")

    def test_en_passant_square_is_part_of_position(self) -> None:
        # The first four fields are compared, so the en passant square counts.
        a = "8/8/8/8/4P3/8/8/8 w - e3 0 1"
        b = "8/8/8/8/4P3/8/8/8 w - - 0 1"
        assert not positions_equal(a, b)
