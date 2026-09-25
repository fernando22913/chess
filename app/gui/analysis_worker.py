from __future__ import annotations

import chess
from PySide6.QtCore import QThread, Signal

from app.engine.stockfish_engine import (
    EngineConfig,
    EngineManager,
)


class AnalysisWorker(QThread):
    """Runs Stockfish analysis in a background thread."""

    started = Signal()
    finished = Signal(list)
    error = Signal(str)
    status_changed = Signal(str)

    def __init__(self, config: EngineConfig | None = None, parent=None) -> None:
        super().__init__(parent)
        self._config = config or EngineConfig()
        self._engine: EngineManager | None = None
        self._board: chess.Board | None = None
        self._stop_requested = False

    @property
    def engine_manager(self) -> EngineManager | None:
        return self._engine

    def start_engine(self) -> str:
        """Start the Stockfish engine. Returns engine name."""
        self._engine = EngineManager(self._config)
        name = self._engine.start()
        self.status_changed.emit("Idle")
        return name

    def request_analysis(self, board: chess.Board) -> None:
        """Request analysis of a position. Stops any previous analysis first."""
        if self._engine is None:
            self.error.emit("Engine not started.")
            return

        if self.isRunning():
            self._stop_requested = True
            self._engine.stop()
            self.wait(5000)

        self._board = board.copy()
        self._stop_requested = False
        self.start()

    def run(self) -> None:
        """Execute analysis in background thread."""
        if self._engine is None or self._board is None:
            self.error.emit("No engine or board available.")
            return

        self.started.emit()
        self.status_changed.emit("Analyzing")

        try:
            results = self._engine.analyse(self._board)
            if not self._stop_requested:
                # Status before finished: the finished slot reads the engine
                # status and must already see "Idle" when it renders.
                self.status_changed.emit("Idle")
                self.finished.emit(results)
        except Exception as e:
            if not self._stop_requested:
                self.status_changed.emit("Error")
                self.error.emit(str(e))

    def stop(self) -> None:
        """Stop ongoing analysis."""
        self._stop_requested = True
        if self._engine:
            self.status_changed.emit("Stopping")
            self._engine.stop()

    def shutdown_engine(self) -> None:
        """Shutdown the engine cleanly."""
        if self.isRunning():
            self.stop()
            self.wait(5000)
        if self._engine:
            self._engine.shutdown()
            self._engine = None
        self.status_changed.emit("Off")
