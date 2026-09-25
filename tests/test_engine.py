import chess
import pytest

from app.engine.stockfish_engine import EngineManager, EngineConfig


@pytest.fixture
def engine():
    """Create and start an EngineManager for testing."""
    config = EngineConfig()
    mgr = EngineManager(config)
    mgr.start()
    yield mgr
    mgr.shutdown()


@pytest.fixture
def starting_board():
    return chess.Board(chess.STARTING_FEN)


class TestEngineStartup:
    def test_engine_starts(self, engine: EngineManager) -> None:
        assert engine.is_running()

    def test_engine_configures(self, engine: EngineManager) -> None:
        assert engine.config.threads == 4
        assert engine.config.hash_mb == 512


class TestAnalysis:
    def test_starting_position_returns_results(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_multipv_returns_three_results(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        assert len(results) == 3

    def test_multipv_indices_are_correct(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        indices = [r.multipv for r in results]
        assert indices == [1, 2, 3]

    def test_scores_are_valid(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        for r in results:
            if r.is_mate:
                assert r.score_mate is not None
                assert r.score_cp is None
            else:
                assert r.score_cp is not None
                assert r.score_mate is None
                assert isinstance(r.score_cp, int)

    def test_depth_is_positive(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        for r in results:
            assert r.depth > 0

    def test_pv_is_non_empty(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        for r in results:
            assert len(r.pv_uci) > 0
            assert len(r.pv_san) > 0

    def test_best_move_san_is_non_empty(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        for r in results:
            assert len(r.best_move_san) > 0

    def test_san_moves_are_valid(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        for r in results:
            temp_board = starting_board.copy()
            for san in r.pv_san:
                move = temp_board.parse_san(san)
                assert move is not None
                temp_board.push(move)

    def test_score_is_from_the_side_to_move(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        # Starting position: White moves, so the POV flag must say White.
        results = engine.analyse(starting_board)
        for r in results:
            assert r.side_to_move_white is True

    def test_time_is_positive(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        for r in results:
            assert r.time_seconds > 0

    def test_time_limit_respected(
        self, engine: EngineManager, starting_board: chess.Board
    ) -> None:
        results = engine.analyse(starting_board)
        for r in results:
            assert r.time_seconds <= 35.0


class TestEngineReuse:
    def test_second_analysis_after_first(
        self, engine: EngineManager
    ) -> None:
        board1 = chess.Board(chess.STARTING_FEN)
        results1 = engine.analyse(board1)
        assert len(results1) == 3

        board2 = chess.Board(
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        )
        results2 = engine.analyse(board2)
        assert len(results2) == 3
        assert results2[0].best_move_san != ""
        # It is Black to move, so every score is now read from Black's side.
        for r in results2:
            assert r.side_to_move_white is False


class TestFenValidation:
    def test_invalid_board_rejected(self, engine: EngineManager) -> None:
        with pytest.raises(ValueError):
            engine.analyse(chess.Board("invalid"))


class TestShutdown:
    def test_shutdown(self) -> None:
        config = EngineConfig()
        mgr = EngineManager(config)
        mgr.start()
        assert mgr.is_running()
        mgr.shutdown()
        assert not mgr.is_running()

    def test_context_manager(self) -> None:
        config = EngineConfig()
        with EngineManager(config) as mgr:
            assert mgr.is_running()
        assert not mgr.is_running()
