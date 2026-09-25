"""Piece-map -> FEN conversion and position comparison.

Bridges recognition and analysis: the 64 symbols produced by
``PieceRecognizer`` (ordered a8..h1) become a FEN string that python-chess and
Stockfish can consume, and two FENs can be compared by the watch loop so the
engine is not restarted for an unchanged position.

Chess-logic validation of the resulting position lives in Phase 7. This module
only converts and compares.
"""

from __future__ import annotations

from typing import Sequence

EMPTY_SQUARE = "."
DEFAULT_SIDE_TO_MOVE = "w"

# Placement, side to move, castling rights and en passant. Move counters are
# deliberately excluded: they change on every ply without the picture changing.
COMPARISON_FIELDS = 4

# Indices of the squares that grant castling rights, in a8..h1 order.
# rank 8 starts at index 0, rank 1 at index 56.
_A8, _E8, _H8 = 0, 4, 7
_A1, _E1, _H1 = 56, 60, 63


class PositionError(ValueError):
    """Raised when a piece map cannot be converted into a FEN."""


def piece_map_to_placement(squares: Sequence[str]) -> str:
    """Convert 64 symbols (a8..h1) into a FEN piece-placement field."""
    if len(squares) != 64:
        raise PositionError(f"Expected 64 squares, got {len(squares)}")

    ranks = []
    for row in range(8):
        rank = ""
        empty = 0
        for col in range(8):
            symbol = squares[row * 8 + col]
            if symbol in (EMPTY_SQUARE, "", None):
                empty += 1
                continue
            if len(symbol) != 1:
                raise PositionError(f"Invalid piece symbol: {symbol!r}")
            if empty:
                rank += str(empty)
                empty = 0
            rank += symbol
        if empty:
            rank += str(empty)
        ranks.append(rank)

    return "/".join(ranks)


def castling_rights(squares: Sequence[str]) -> str:
    """Derive KQkq rights from the piece map.

    Rights are granted when the king and at least one rook still stand on their
    original squares. Without a move history this is a heuristic: it cannot know
    whether the king has already moved away and come back.
    """
    if len(squares) != 64:
        raise PositionError(f"Expected 64 squares, got {len(squares)}")

    # Canonical FEN order: K Q k q.
    rights = ""
    if squares[_E1] == "K" and squares[_H1] == "R":
        rights += "K"
    if squares[_E1] == "K" and squares[_A1] == "R":
        rights += "Q"
    if squares[_E8] == "k" and squares[_H8] == "r":
        rights += "k"
    if squares[_E8] == "k" and squares[_A8] == "r":
        rights += "q"

    return rights or "-"


def piece_map_to_fen(
    squares: Sequence[str],
    side_to_move: str = DEFAULT_SIDE_TO_MOVE,
) -> str:
    """Build a complete FEN from 64 symbols (a8..h1)."""
    if side_to_move not in ("w", "b"):
        raise PositionError(f"Invalid side to move: {side_to_move!r}")

    placement = piece_map_to_placement(squares)
    return f"{placement} {side_to_move} {castling_rights(squares)} - 0 1"


def normalize_fen(fen: str) -> str:
    """Reduce a FEN to the fields that identify the position itself."""
    fields = fen.split()
    if len(fields) < COMPARISON_FIELDS:
        raise PositionError(f"FEN has too few fields: {fen!r}")
    return " ".join(fields[:COMPARISON_FIELDS])


def positions_equal(fen_a: str, fen_b: str) -> bool:
    """True when two FENs describe the same position.

    Move counters are ignored, so the same position with a different halfmove
    clock still compares equal.
    """
    return normalize_fen(fen_a) == normalize_fen(fen_b)
