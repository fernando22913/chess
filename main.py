#!/usr/bin/env python3
"""Stockfish Vision Trainer — Main Entry Point."""

import sys

from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Stockfish Vision Trainer")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
