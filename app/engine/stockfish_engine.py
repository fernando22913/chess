from __future__ import annotations

from dataclasses import dataclass

import chess
import chess.engine

STOCKFISH_PATH = "/usr/games/stockfish"
DEFAULT_THREADS = 4
DEFAULT_HASH_MB = 512
DEFAULT_MULTIPV = 3
DEFAULT_DEPTH = 22
DEFAULT_TIME_LIMIT = 30.0


@dataclass
class EngineConfig:
    stockfish_path: str = STOCKFISH_PATH
    threads: int = DEFAULT_THREADS
    hash_mb: int = DEFAULT_HASH_MB
    multipv: int = DEFAULT_MULTIPV
    target_depth: int = DEFAULT_DEPTH
    time_limit_seconds: float = DEFAULT_TIME_LIMIT


@dataclass
class AnalysisResult:
    multipv: int
    score_cp: int | None
    score_mate: int | None
    depth: int
    seldepth: int | None
    nodes: int | None
    time_seconds: float
    pv_uci: list[str]
    pv_san: list[str]
    best_move_san: str
    is_mate: bool
    side_to_move_white: bool
    """Score POV: score_cp/score_mate are always from the side to move."""


class EngineManager:
    def __init__(self, config: EngineConfig | None = None) -> None:
        self._config = config or EngineConfig()
        self._engine: chess.engine.SimpleEngine | None = None

    @property
    def config(self) -> EngineConfig:
        return self._config

    def is_running(self) -> bool:
        return self._engine is not None

    def start(self) -> str:
        """Start Stockfish and return engine id string."""
        self._engine = chess.engine.SimpleEngine.popen_uci(
            self._config.stockfish_path
        )
        self._engine.configure({
            "Threads": self._config.threads,
            "Hash": self._config.hash_mb,
        })
        return self._engine.id.get("name", "Unknown Engine")

    def analyse(self, board: chess.Board) -> list[AnalysisResult]:
        """Blocking analysis with MultiPV. Returns list of AnalysisResult."""
        if self._engine is None:
            raise RuntimeError("Engine not started. Call start() first.")

        if not board.is_valid():
            raise ValueError(f"Invalid board position: {board.fen()}")

        limit = chess.engine.Limit(
            depth=self._config.target_depth,
            time=self._config.time_limit_seconds,
        )

        info_list = self._engine.analyse(
            board, limit, multipv=self._config.multipv
        )

        return [self._build_result(board, info) for info in info_list]

    def _build_result(self, board: chess.Board, info: dict) -> AnalysisResult:
        # Phase 8: the evaluation must be read from the point of view of the
        # side to move, not always from White. pov() negates the relative
        # score when the requested colour differs from the root turn.
        score = info["score"].pov(board.turn)

        if score.is_mate():
            score_mate = score.mate()
            score_cp = None
            is_mate = True
        else:
            score_cp = score.score()
            score_mate = None
            is_mate = False

        pv_moves = info.get("pv", [])
        pv_uci = [str(m) for m in pv_moves]
        pv_san = _pv_to_san(board, pv_moves)
        best_move_san = pv_san[0] if pv_san else ""

        return AnalysisResult(
            multipv=info.get("multipv", 1),
            score_cp=score_cp,
            score_mate=score_mate,
            depth=info.get("depth", 0),
            seldepth=info.get("seldepth"),
            nodes=info.get("nodes"),
            time_seconds=info.get("time", 0.0),
            pv_uci=pv_uci,
            pv_san=pv_san,
            best_move_san=best_move_san,
            is_mate=is_mate,
            side_to_move_white=board.turn == chess.WHITE,
        )

    def stop(self) -> None:
        """Stop ongoing analysis. Intended for Phase 2 worker thread use."""
        if self._engine:
            self._engine.stop()

    def shutdown(self) -> None:
        """Quit Stockfish cleanly."""
        if self._engine:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> EngineManager:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()


def _pv_to_san(board: chess.Board, pv_moves: list[chess.Move]) -> list[str]:
    """Convert a list of UCI moves to SAN using board context."""
    temp_board = board.copy()
    san_moves = []
    for move in pv_moves:
        san_moves.append(temp_board.san(move))
        temp_board.push(move)
    return san_moves
