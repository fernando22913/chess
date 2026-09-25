"""Phase 9: human explanation built strictly from objective position facts.

The specification is explicit about three things:

* Stockfish must not be presented as if it generated human explanations by
  itself, and the real engine information must always be shown first.
* Any explanation must be based on objective information from the position.
* Do not invent explanations.

So every sentence produced here is derived from the FEN and from the engine's
chosen move through python-chess itself: material counts, check, castling
rights, squares occupied, what the move physically does.  Judgements such as
"controls the centre", "is a strong move" or "prepares castling" are
deliberately out of scope, because nothing in the position proves them.
"""

from __future__ import annotations

import chess

EXPLANATION_HEADER = "HUMAN EXPLANATION"

DISCLAIMER = (
    "Objective facts computed from the position itself. "
    "Stockfish supplies only the move and the evaluation."
)

_PIECE_POINTS = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

_PIECE_ORDER = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
)

_MINOR_PIECES = (chess.KNIGHT, chess.BISHOP)

_CENTRE_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)


def _side_name(color: bool) -> str:
    return "White" if color == chess.WHITE else "Black"


def _opposite(color: bool) -> bool:
    return not color


def _count(piece_type: int, amount: int) -> str:
    """``queen`` / ``2 pawns``: the number only matters when there is more."""
    label = chess.piece_name(piece_type)
    if amount == 1:
        return label
    return f"{amount} {label}s"


def _material_lines(board: chess.Board) -> list[str]:
    counts = {chess.WHITE: {}, chess.BLACK: {}}
    points = {chess.WHITE: 0, chess.BLACK: 0}
    for piece in board.piece_map().values():
        counts[piece.color][piece.piece_type] = (
            counts[piece.color].get(piece.piece_type, 0) + 1
        )
        points[piece.color] += _PIECE_POINTS[piece.piece_type]

    if points[chess.WHITE] == points[chess.BLACK]:
        return ["Material is equal."]

    leader = chess.WHITE if points[chess.WHITE] > points[chess.BLACK] else chess.BLACK
    lead = abs(points[chess.WHITE] - points[chess.BLACK])
    surplus = [
        _count(piece_type, counts[leader][piece_type] - counts[_opposite(leader)].get(piece_type, 0))
        for piece_type in _PIECE_ORDER
        if counts[leader].get(piece_type, 0) > counts[_opposite(leader)].get(piece_type, 0)
    ]
    line = f"{_side_name(leader)} is ahead by {lead} point" f"{'' if lead == 1 else 's'}"
    if surplus:
        line += f" ({', '.join(surplus)})"
    return [line + "."]


def _check_lines(board: chess.Board) -> list[str]:
    if not board.is_check():
        return []
    return [f"{_side_name(board.turn)} king is in check."]


def _castling_lines(board: chess.Board) -> list[str]:
    lines = []
    for color in (chess.WHITE, chess.BLACK):
        kingside = board.has_kingside_castling_rights(color)
        queenside = board.has_queenside_castling_rights(color)
        if kingside and queenside:
            lines.append(f"{_side_name(color)} may castle on both sides.")
        elif kingside:
            lines.append(f"{_side_name(color)} may castle kingside.")
        elif queenside:
            lines.append(f"{_side_name(color)} may castle queenside.")
    if not lines:
        lines.append("Neither side has castling rights.")
    return lines


def _back_rank_counts(board: chess.Board, color: bool) -> tuple[int, int]:
    """``(on the back rank, total)`` for that side's knights and bishops."""
    home_rank = 0 if color == chess.WHITE else 7
    on_back = 0
    total = 0
    for piece_type in _MINOR_PIECES:
        for square in board.pieces(piece_type, color):
            total += 1
            if chess.square_rank(square) == home_rank:
                on_back += 1
    return on_back, total


def _development_lines(board: chess.Board) -> list[str]:
    per_side = {color: _back_rank_counts(board, color) for color in (chess.WHITE, chess.BLACK)}
    if all(on_back == total for on_back, total in per_side.values()):
        return []
    return [
        f"{_side_name(color)} has {on_back} of {total} knights and bishops on the back rank."
        for color, (on_back, total) in per_side.items()
    ]


def _centre_lines(board: chess.Board) -> list[str]:
    occupied = []
    for square in _CENTRE_SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            occupied.append(
                f"{_side_name(piece.color)} {chess.piece_name(piece.piece_type)} "
                f"on {chess.square_name(square)}"
            )
    if not occupied:
        return ["None of d4, e4, d5 or e5 is occupied."]
    return [f"On the centre squares: {', '.join(occupied)}."]


def _best_move_lines(board: chess.Board, move: chess.Move) -> list[str]:
    """What the move *does*, never what it *means*."""
    san = board.san(move)
    after = board.copy(stack=False)
    after.push(move)

    lines = []

    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) == 6 else "queenside"
        lines.append(f"The best move {san} castles {side}.")

    if board.is_capture(move):
        if board.is_en_passant(move):
            takes = "a pawn (en passant)"
        else:
            taken = board.piece_at(move.to_square)
            takes = f"a {chess.piece_name(taken.piece_type)}" if taken is not None else "a piece"
        lines.append(f"The best move {san} captures {takes}.")

    if move.promotion is not None:
        lines.append(
            f"The best move {san} promotes to a {chess.piece_name(move.promotion)}."
        )

    if board.gives_check(move):
        verdict = "gives mate" if after.is_checkmate() else "gives check"
        lines.append(f"The best move {san} {verdict}.")

    if not lines:
        lines.append(
            f"The best move {san} neither captures nor gives check."
        )

    if not after.is_check() and not any(after.legal_moves):
        lines.append("After the best move the opponent has no legal move (stalemate).")

    return lines


def explain_position(
    board: chess.Board,
    best_move: chess.Move | None = None,
) -> list[str]:
    """Objective observations about *board*, engine facts excluded.

    The engine has already reported the move, the score and the variation;
    this only adds what can be read straight off the position.
    """
    lines: list[str] = []
    if best_move is not None and board.is_legal(best_move):
        lines.extend(_best_move_lines(board, best_move))
    lines.extend(_material_lines(board))
    lines.extend(_check_lines(board))
    lines.extend(_castling_lines(board))
    lines.extend(_development_lines(board))
    lines.extend(_centre_lines(board))
    return lines


def format_explanation(lines: list[str]) -> str:
    """Render the block, header and disclaimer included."""
    if not lines:
        return ""
    return "\n".join([EXPLANATION_HEADER, DISCLAIMER, ""] + lines)


def explain_fen(fen: str, best_move_uci: str | None = None) -> str:
    """Convenience entry point used by the GUI: FEN in, block out."""
    try:
        board = chess.Board(fen)
    except ValueError:
        return ""

    move: chess.Move | None = None
    if best_move_uci:
        try:
            candidate = chess.Move.from_uci(best_move_uci)
        except ValueError:
            candidate = None
        if candidate is not None and board.is_legal(candidate):
            move = candidate

    return format_explanation(explain_position(board, best_move=move))
