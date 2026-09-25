from __future__ import annotations

import chess
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

PIECE_UNICODE = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656", "B": "\u2657", "N": "\u2658", "P": "\u2659",
    "k": "\u265A", "q": "\u265B", "r": "\u265C", "b": "\u265D", "n": "\u265E", "p": "\u265F",
}


def board_to_unicode(board: chess.Board) -> str:
    """Convert board to Unicode piece representation."""
    lines = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if piece:
                row.append(PIECE_UNICODE[piece.symbol()])
            else:
                row.append("\u2003")
        lines.append(" ".join(row))
    return "\n".join(lines)


class BoardWidget(QWidget):
    """Widget that displays the current chess position."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("BOARD")
        group_layout = QVBoxLayout()

        self._board_display = QTextEdit()
        self._board_display.setReadOnly(True)
        font = QFont("Monospace", 16)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._board_display.setFont(font)
        self._board_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._board_display.setMaximumHeight(260)
        self._board_display.setPlaceholderText("No position loaded.\nPaste a FEN or use Detect to load a position.")
        group_layout.addWidget(self._board_display)

        self._fen_label = QLabel("FEN: —")
        self._fen_label.setWordWrap(True)
        self._fen_label.setStyleSheet("color: #666; font-size: 11px;")
        group_layout.addWidget(self._fen_label)

        group.setLayout(group_layout)
        layout.addWidget(group)

        self.clear()

    def set_position(self, board: chess.Board) -> None:
        """Display a chess position."""
        self._board_display.setPlainText(board_to_unicode(board))
        self._fen_label.setText(f"FEN: {board.fen()}")

    def set_fen(self, fen: str) -> None:
        """Display a position from FEN string."""
        board = chess.Board(fen)
        self.set_position(board)

    def clear(self) -> None:
        self._board_display.clear()
        self._fen_label.setText("FEN: —")
