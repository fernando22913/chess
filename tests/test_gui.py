import os
import threading
import time
import chess
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.engine.stockfish_engine import EngineConfig
from app.gui.board_widget import BoardWidget, board_to_unicode
from app.gui.main_window import MainWindow
from app.gui.analysis_worker import AnalysisWorker
from tests.test_analysis_formatter import make_result


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _close_main_windows(qapp):
    """Shut down engines opened by MainWindow so tests never leak Stockfish."""
    yield
    for widget in qapp.topLevelWidgets():
        if isinstance(widget, MainWindow):
            widget.close()


def spin_until(predicate, timeout: float) -> bool:
    """Run the Qt event loop until *predicate* holds or *timeout* expires.

    Cross-thread signals are queued, so they are only delivered while the
    event loop is actually turning.
    """
    deadline = time.monotonic() + timeout
    app = QApplication.instance()
    while time.monotonic() < deadline:
        if predicate():
            return True
        app.processEvents()
        time.sleep(0.01)
    return predicate()


class TestAnalysisDisplay:
    """Phase 8 in the GUI: the result block shows what the spec asks for."""

    def test_results_reach_the_analysis_display(self, qapp) -> None:
        window = MainWindow()
        window._display_results([make_result()])

        text = window._analysis_display.toPlainText()
        assert "BEST MOVE\nNf3\n+0.84" in text
        assert "PRINCIPAL VARIATION\nNf3" in text
        for label in (
            "Depth:",
            "Nodes:",
            "Analysis time:",
            "Evaluation:",
            "Side to move:",
            "MultiPV:",
            "Engine status:",
        ):
            assert label in text

    def test_alternatives_are_listed_under_their_own_header(
        self, qapp
    ) -> None:
        window = MainWindow()
        window._display_results(
            [
                make_result(),
                make_result(multipv=2, best_move_san="d4", score_cp=72),
            ]
        )

        text = window._analysis_display.toPlainText()
        assert "ALTERNATIVES" in text
        assert "2. d4" in text
        assert "   +0.72" in text

    def test_engine_status_is_the_idle_one_after_a_finished_analysis(
        self, qapp
    ) -> None:
        window = MainWindow()
        window._engine_status = "Idle"
        window._display_results([make_result()])

        assert "Engine status: Idle" in (
            window._analysis_display.toPlainText()
        )

    def test_engine_name_is_shown(self, qapp) -> None:
        window = MainWindow()
        window._display_results([make_result()])
        text = window._analysis_display.toPlainText()
        assert "Engine:" in text
        assert "Stockfish" in text

    def test_engine_labels_follow_the_results(self, qapp) -> None:
        window = MainWindow()
        window._display_results(
            [make_result(depth=22, nodes=1_234_567, time_seconds=5.5)]
        )
        assert window._depth_label.text() == "Depth: 22"
        assert window._time_label.text() == "Time: 5.5s"

    def test_empty_results_are_reported(self, qapp) -> None:
        window = MainWindow()
        window._display_results([])
        assert window._analysis_display.toPlainText() == "No results."

    def test_multi_pv_is_shown_in_the_engine_panel(self, qapp) -> None:
        window = MainWindow()
        assert "MultiPV: 3" in window._multipv_label.text()

    def test_engine_status_survives_a_status_change(self, qapp) -> None:
        window = MainWindow()
        window._on_status_changed("Analyzing")
        assert window._engine_status == "Analyzing"
        assert "Engine:" in window._status_label.text()


class TestHumanExplanation:
    """Phase 9 in the GUI: engine facts first, objective notes after them."""

    @staticmethod
    def _fill(window) -> None:
        window._last_fen = chess.STARTING_FEN
        window._engine_status = "Idle"
        window._engine_name = "Stockfish 17.1"

    def test_the_explanation_comes_after_the_engine_information(
        self, qapp
    ) -> None:
        window = MainWindow()
        self._fill(window)
        window._display_results([make_result()])

        text = window._analysis_display.toPlainText()
        assert "--- ENGINE DATA ---" in text
        assert "HUMAN EXPLANATION" in text
        assert text.index("--- ENGINE DATA ---") < text.index("HUMAN EXPLANATION")

    def test_it_says_that_stockfish_only_brings_move_and_score(
        self, qapp
    ) -> None:
        window = MainWindow()
        self._fill(window)
        window._display_results([make_result()])

        text = window._analysis_display.toPlainText()
        assert "Stockfish supplies only the move and the evaluation." in text
        assert "Material is equal." in text

    def test_no_results_means_no_explanation(self, qapp) -> None:
        window = MainWindow()
        self._fill(window)
        window._display_results([])

        assert "HUMAN EXPLANATION" not in (
            window._analysis_display.toPlainText()
        )

    def test_an_unknown_position_gets_no_explanation(self, qapp) -> None:
        window = MainWindow()
        window._engine_status = "Idle"
        window._last_fen = None
        window._display_results([make_result()])

        assert "HUMAN EXPLANATION" not in (
            window._analysis_display.toPlainText()
        )
        assert "BEST MOVE" in window._analysis_display.toPlainText()


class TestBoardToUnicode:
    def test_starting_position(self) -> None:
        board = chess.Board(chess.STARTING_FEN)
        result = board_to_unicode(board)
        assert "\u2654" in result  # white king
        assert "\u265A" in result  # black king
        assert "\u2659" in result  # white pawn
        assert "\u265F" in result  # black pawn

    def test_empty_board(self) -> None:
        board = chess.Board()
        board.clear()
        result = board_to_unicode(board)
        assert "\u2654" not in result
        assert "\u265A" not in result


class TestBoardWidget:
    def test_creation(self, qapp) -> None:
        widget = BoardWidget()
        assert widget is not None

    def test_set_position(self, qapp) -> None:
        widget = BoardWidget()
        board = chess.Board(chess.STARTING_FEN)
        widget.set_position(board)
        assert widget._fen_label.text() == f"FEN: {chess.STARTING_FEN}"

    def test_set_fen(self, qapp) -> None:
        widget = BoardWidget()
        fen = chess.STARTING_FEN
        widget.set_fen(fen)
        assert fen in widget._fen_label.text()

    def test_clear(self, qapp) -> None:
        widget = BoardWidget()
        widget.set_fen(chess.STARTING_FEN)
        widget.clear()
        assert widget._fen_label.text() == "FEN: —"


class TestMainWindow:
    def test_creation(self, qapp) -> None:
        window = MainWindow()
        assert window.windowTitle() == "Stockfish Vision Trainer"

    def test_engine_started(self, qapp) -> None:
        window = MainWindow()
        status = window._status_label.text()
        assert "Idle" in status

    def test_threads_displayed(self, qapp) -> None:
        window = MainWindow()
        assert "Threads: 4" in window._threads_label.text()

    def test_hash_displayed(self, qapp) -> None:
        window = MainWindow()
        assert "Hash: 512 MB" in window._hash_label.text()


class TestWatch:
    def test_watch_disabled_until_the_board_is_detected(self, qapp) -> None:
        window = MainWindow()
        assert window._watch_btn.isEnabled() is False

    def test_start_watch_without_a_detected_board_is_refused(self, qapp) -> None:
        window = MainWindow()
        window._start_watch()
        assert window._watcher is None
        assert "Select a region and detect the board first" in (
            window._analysis_display.toPlainText()
        )

    def test_stop_watch_is_safe_when_idle(self, qapp) -> None:
        window = MainWindow()
        window._stop_watch()
        assert window._watcher is None
        assert window._watch_btn.text() == "Watch"

    def test_unchanged_position_does_not_restart_the_engine(
        self, qapp, monkeypatch
    ) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        window._last_fen = chess.STARTING_FEN
        same_position = chess.STARTING_FEN.replace("0 1", "99 42")
        window._on_watched_position(["."] * 64, same_position)

        assert calls == [], "an identical position must not run Stockfish again"

    def test_changed_position_starts_an_analysis(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        window._last_fen = chess.STARTING_FEN
        board = chess.Board(chess.STARTING_FEN)
        board.push_uci("e2e4")
        window._on_watched_position(["."] * 64, board.fen())

        assert len(calls) == 1
        assert window._last_fen == board.fen()

    def test_first_watched_position_is_accepted_without_a_baseline(
        self, qapp, monkeypatch
    ) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        window._last_fen = None
        window._on_watched_position(["."] * 64, chess.STARTING_FEN)

        assert len(calls) == 1

    def test_unparseable_position_is_rejected(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        window._last_fen = chess.STARTING_FEN
        window._on_watched_position(["."] * 64, "not a fen at all")

        assert calls == []
        assert window._last_fen == chess.STARTING_FEN

    def test_position_without_kings_is_not_sent_to_the_engine(
        self, qapp, monkeypatch
    ) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        window._last_fen = chess.STARTING_FEN
        window._on_watched_position(["."] * 64, "8/8/8/8/8/8/8/8 w - - 0 1")

        assert calls == []


class TestPositionValidation:
    """Phase 7 in the GUI: an impossible position never reaches Stockfish."""

    @staticmethod
    def _starting_squares() -> list[str]:
        board = chess.Board(chess.STARTING_FEN)
        return [
            piece.symbol() if (piece := board.piece_at(chess.square(f, r))) else "."
            for r in range(7, -1, -1)
            for f in range(8)
        ]

    def test_valid_fen_reaches_the_engine(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        assert window._analyze_position(chess.STARTING_FEN) is True
        assert len(calls) == 1
        assert window._last_fen == chess.STARTING_FEN

    @pytest.mark.parametrize(
        "fen",
        ["not a fen", "", "8/8/8/8/8/8/8/8 w - - 0 1", "4k3/8/8/8/8/8/8/P3K3 w - - 0 1"],
    )
    def test_invalid_fen_never_reaches_the_engine(
        self, qapp, monkeypatch, fen: str
    ) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)
        window._last_fen = chess.STARTING_FEN

        assert window._analyze_position(fen) is False
        assert calls == []
        assert window._last_fen == chess.STARTING_FEN

    def test_the_user_is_told_why(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        monkeypatch.setattr(window._worker, "request_analysis", lambda _b: None)

        window._analyze_position("not a fen")

        text = window._analysis_display.toPlainText()
        assert "Position invalid. Rechecking recognition..." in text
        assert "turn" in text

    def test_typed_fen_uses_the_same_gate(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        window._fen_input.setText("8/8/8/8/8/8/8/8 w - - 0 1")
        window._on_fen_submitted()

        assert calls == []

    def test_typed_valid_fen_is_analyzed(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)

        window._fen_input.setText(chess.STARTING_FEN)
        window._on_fen_submitted()

        assert len(calls) == 1

    def test_garbage_recognition_does_not_enable_watch(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)
        monkeypatch.setattr(
            window._recognizer,
            "recognize_board_detailed",
            lambda _board: (["."] * 64, [0.0] * 64),
        )

        window._recognize_pieces(None)  # type: ignore[arg-type]

        assert calls == []
        assert window._watch_btn.isEnabled() is False
        assert "Position invalid" in window._analysis_display.toPlainText()

    def test_good_recognition_enables_watch(self, qapp, monkeypatch) -> None:
        window = MainWindow()
        calls: list = []
        monkeypatch.setattr(window._worker, "request_analysis", calls.append)
        squares = self._starting_squares()
        confidences = [0.95 if s != "." else 0.0 for s in squares]
        monkeypatch.setattr(
            window._recognizer,
            "recognize_board_detailed",
            lambda _board: (list(squares), confidences),
        )

        window._recognize_pieces(None)  # type: ignore[arg-type]

        assert len(calls) == 1
        assert window._watch_btn.isEnabled() is True
        assert window._fen_input.text() == chess.STARTING_FEN


class TestAnalysisWorker:
    # A deliberately small engine config: these tests check the plumbing,
    # not Stockfish strength, and must not hog the machine.
    LIGHT = EngineConfig(threads=1, hash_mb=16, target_depth=6,
                         time_limit_seconds=5.0)

    def test_creation(self) -> None:
        worker = AnalysisWorker()
        assert worker is not None

    def test_start_engine(self) -> None:
        worker = AnalysisWorker(self.LIGHT)
        name = worker.start_engine()
        assert "Stockfish" in name
        worker.shutdown_engine()

    def test_request_analysis(self) -> None:
        worker = AnalysisWorker(self.LIGHT)
        worker.start_engine()
        results: list = []
        worker.finished.connect(results.extend)

        worker.request_analysis(chess.Board(chess.STARTING_FEN))
        spin_until(lambda: bool(results), 30.0)
        worker.shutdown_engine()

        assert results, "the analysis never delivered a result"
        assert results[0].best_move_san


class TestAnalysisCancellation:
    """Spec: a new position must cancel the analysis in progress.

    The search itself is stubbed out: this test exercises our own
    cancellation logic and finishes in well under a second, so it never
    competes with the desktop for CPU.
    """

    def test_a_second_request_cancels_the_first(self, qapp, monkeypatch) -> None:
        # A tiny engine config: the process is only opened to satisfy the
        # worker, the actual search is replaced below.
        worker = AnalysisWorker(
            EngineConfig(threads=1, hash_mb=16, target_depth=1)
        )
        worker.start_engine()
        manager = worker.engine_manager

        search_started = threading.Event()
        allow_search_to_return = threading.Event()
        calls: list[chess.Board] = []

        def blocking_analyse(board: chess.Board) -> list:
            calls.append(board.copy())
            search_started.set()
            if not allow_search_to_return.wait(10.0):
                raise AssertionError("the analysis was never cancelled")
            # Answer for the board that was actually handed to the engine,
            # so the test can tell the two analyses apart afterwards.
            return [make_result(side_to_move_white=board.turn == chess.WHITE)]

        def fake_stop() -> None:
            allow_search_to_return.set()

        monkeypatch.setattr(manager, "analyse", blocking_analyse)
        monkeypatch.setattr(manager, "stop", fake_stop)

        finished: list = []
        errors: list = []
        worker.finished.connect(finished.append)
        worker.error.connect(errors.append)

        first = chess.Board(chess.STARTING_FEN)  # White to move
        second = chess.Board(chess.STARTING_FEN)  # Black to move
        second.push_uci("e2e4")

        worker.request_analysis(first)
        assert spin_until(search_started.is_set, 5.0), (
            "the first analysis never started"
        )

        # A new valid position arrives while the first is still searching:
        # it must cancel the previous one and deliver only the new results.
        worker.request_analysis(second)
        assert spin_until(lambda: len(calls) == 2, 5.0), (
            "the second analysis never started"
        )
        assert spin_until(lambda: bool(finished), 5.0), (
            "the second analysis never finished"
        )
        worker.shutdown_engine()

        assert errors == []
        assert len(finished) == 1, "the cancelled analysis leaked a result"
        assert finished[0][0].side_to_move_white is False, (
            "the delivered results must belong to the second position"
        )
