from __future__ import annotations

import cv2
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont, QPixmap, QImage
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.capture.screen_capture import ScreenRegion, capture_region
from app.capture.region_selector import RegionSelector
from app.capture.board_watcher import BoardWatcher, WatchConfig
from app.chess.position import piece_map_to_fen, positions_equal
from app.chess.explanation import explain_fen
from app.chess.validation import validate_fen
from app.engine.stockfish_engine import AnalysisResult
from app.gui.analysis_formatter import format_analysis
from app.gui.analysis_worker import AnalysisWorker
from app.gui.board_widget import BoardWidget
from app.vision.board_detector import DetectedBoard, detect_board
from app.vision.piece_recognizer import (
    EMPTY_SQUARE,
    PieceRecognizer,
    PieceRecognizerError,
    format_piece_map,
)


def cv_to_pixmap(image) -> QPixmap:
    """Convert an OpenCV BGR image to QPixmap."""
    if image is None:
        return QPixmap()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Stockfish Vision Trainer")
        self.setMinimumSize(900, 650)

        self._selected_region: ScreenRegion | None = None
        self._region_selector: RegionSelector | None = None
        self._detected_board: DetectedBoard | None = None
        self._recognizer = PieceRecognizer()
        self._watcher: BoardWatcher | None = None
        self._last_fen: str | None = None
        self._engine_name = ""
        self._engine_status = "Off"

        self._worker = AnalysisWorker(parent=self)
        self._worker.started.connect(self._on_analysis_started)
        self._worker.finished.connect(self._on_analysis_finished)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.status_changed.connect(self._on_status_changed)

        self._setup_ui()
        self._start_engine()

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 4, 4, 4)

        self._board_widget = BoardWidget()
        left_layout.addWidget(self._board_widget)

        self._fen_input = QLineEdit()
        self._fen_input.setPlaceholderText("Paste FEN here and press Enter...")
        self._fen_input.returnPressed.connect(self._on_fen_submitted)
        left_layout.addWidget(self._fen_input)

        self._capture_preview = QLabel("No capture yet.")
        self._capture_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._capture_preview.setMinimumHeight(100)
        self._capture_preview.setStyleSheet("border: 1px solid #ccc; color: #888;")
        left_layout.addWidget(self._capture_preview)

        splitter.addWidget(left_panel)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._setup_analysis_section(right_layout)
        self._setup_engine_section(right_layout)
        self._setup_controls(right_layout)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _setup_analysis_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("ANALYSIS")
        layout = QVBoxLayout()

        self._analysis_display = QTextEdit()
        self._analysis_display.setReadOnly(True)
        font = QFont("Monospace", 11)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._analysis_display.setFont(font)
        self._analysis_display.setPlaceholderText(
            "No analysis yet.\nEnter a FEN position to start analysis."
        )
        layout.addWidget(self._analysis_display)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _setup_engine_section(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("ENGINE")
        layout = QVBoxLayout()

        self._status_label = QLabel("Status: Off")
        layout.addWidget(self._status_label)

        self._depth_label = QLabel("Depth: —")
        layout.addWidget(self._depth_label)

        self._time_label = QLabel("Time: —")
        layout.addWidget(self._time_label)

        self._cpu_label = QLabel("CPU: —")
        layout.addWidget(self._cpu_label)

        self._ram_label = QLabel("RAM: —")
        layout.addWidget(self._ram_label)

        self._threads_label = QLabel("Threads: —")
        layout.addWidget(self._threads_label)

        self._hash_label = QLabel("Hash: —")
        layout.addWidget(self._hash_label)

        self._multipv_label = QLabel("MultiPV: —")
        layout.addWidget(self._multipv_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _setup_controls(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("CONTROLS")
        layout = QVBoxLayout()

        btn_row1 = QHBoxLayout()
        self._select_region_btn = QPushButton("Select Region")
        self._select_region_btn.setEnabled(True)
        self._select_region_btn.clicked.connect(self._on_select_region)
        btn_row1.addWidget(self._select_region_btn)

        self._detect_btn = QPushButton("Detect")
        self._detect_btn.setEnabled(False)
        self._detect_btn.clicked.connect(self._on_detect)
        btn_row1.addWidget(self._detect_btn)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self._watch_btn = QPushButton("Watch")
        self._watch_btn.setEnabled(False)
        self._watch_btn.clicked.connect(self._on_watch_clicked)
        btn_row2.addWidget(self._watch_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row2.addWidget(self._stop_btn)
        layout.addLayout(btn_row2)

        self._region_label = QLabel("Region: none")
        self._region_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self._region_label)

        group.setLayout(layout)
        parent_layout.addWidget(group)

    def _start_engine(self) -> None:
        try:
            name = self._worker.start_engine()
            config = self._worker.engine_manager.config
            self._engine_name = name
            self._engine_status = "Idle"
            self._threads_label.setText(f"Threads: {config.threads}")
            self._hash_label.setText(f"Hash: {config.hash_mb} MB")
            self._multipv_label.setText(f"MultiPV: {config.multipv}")
            self._status_label.setText(f"Status: Idle\nEngine: {name}")
        except Exception as e:
            self._engine_status = "Error"
            self._status_label.setText(f"Status: Error\n{e}")

    @Slot()
    def _on_select_region(self) -> None:
        self._region_selector = RegionSelector()
        self._region_selector.region_selected.connect(self._on_region_selected)
        self._region_selector.selection_cancelled.connect(self._on_region_cancelled)
        self._region_selector.start_selection()

    @Slot(ScreenRegion)
    def _on_region_selected(self, region: ScreenRegion) -> None:
        self._stop_watch()
        self._selected_region = region
        self._region_label.setText(
            f"Region: ({region.x}, {region.y}) {region.width}x{region.height}"
        )
        self._detect_btn.setEnabled(True)
        self._detected_board = None
        self._last_fen = None

    @Slot()
    def _on_region_cancelled(self) -> None:
        self._region_label.setText("Region: selection cancelled")

    @Slot()
    def _on_detect(self) -> None:
        if self._selected_region is None:
            self._analysis_display.setPlainText("No region selected.")
            return

        self._stop_watch()

        try:
            image = capture_region(self._selected_region)
        except Exception as e:
            self._analysis_display.setPlainText(f"Capture failed:\n{e}")
            return

        pixmap = cv_to_pixmap(image)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self._capture_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._capture_preview.setPixmap(scaled)

        board = detect_board(image)
        if board is None:
            self._analysis_display.setPlainText(
                "Board detection failed.\nPlease select the board again."
            )
            return

        self._detected_board = board

        board_pixmap = cv_to_pixmap(board.transformed)
        if not board_pixmap.isNull():
            scaled = board_pixmap.scaled(
                self._capture_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._capture_preview.setPixmap(scaled)

        self._analysis_display.setPlainText(
            f"Board detected successfully.\n"
            f"Board size: {board.transformed.shape[1]}x{board.transformed.shape[0]}\n"
            f"Squares: {len(board.squares)}\n\n"
            "Board geometry stored.\n"
            "Future captures of this region will reuse it without re-detection."
        )
        self._recognize_pieces(board)

    def _recognize_pieces(self, board: DetectedBoard) -> None:
        try:
            squares, confidences = self._recognizer.recognize_board_detailed(board)
        except PieceRecognizerError as exc:
            self._analysis_display.append(f"\n\nPiece recognition unavailable:\n{exc}")
            return

        occupied = sum(1 for symbol in squares if symbol != EMPTY_SQUARE)
        fen = piece_map_to_fen(squares)
        self._analysis_display.append(
            f"\n\nPieces recognized: {occupied}\n\n"
            f"{format_piece_map(squares)}\n\nFEN:\n{fen}"
        )
        if self._analyze_position(fen, squares=squares, confidences=confidences):
            self._fen_input.setText(fen)
            self._watch_btn.setEnabled(True)

    def _analyze_position(
        self,
        fen: str,
        *,
        squares: list[str] | None = None,
        confidences: list[float] | None = None,
    ) -> bool:
        """Validate a recognized FEN and send it to the engine.

        Phase 7 gates everything here: nothing reaches Stockfish unless it
        parses, holds exactly one king per side and is not obviously
        impossible. Returns True when the position was accepted, in which case
        it becomes the baseline the watch loop compares against.
        """
        result = validate_fen(fen, squares=squares, confidences=confidences)
        if not result.ok:
            self._analysis_display.append("\n\n" + result.display)
            return False

        self._last_fen = fen
        self._board_widget.set_position(result.board)
        self._worker.request_analysis(result.board)
        return True

    @Slot()
    def _on_watch_clicked(self) -> None:
        if self._watcher is not None and self._watcher.isRunning():
            self._stop_watch()
        else:
            self._start_watch()

    def _start_watch(self) -> None:
        if self._selected_region is None or self._detected_board is None:
            self._analysis_display.setPlainText(
                "Select a region and detect the board first."
            )
            return

        self._watcher = BoardWatcher(
            self._selected_region,
            self._detected_board,
            self._recognizer,
            WatchConfig(),
            parent=self,
        )
        self._watcher.position_ready.connect(self._on_watched_position)
        self._watcher.failure.connect(self._on_watch_failure)

        self._detect_btn.setEnabled(False)
        self._select_region_btn.setEnabled(False)
        self._watch_btn.setText("Stop Watch")
        self._set_watch_label(True)
        self._watcher.start()

    def _stop_watch(self) -> None:
        if self._watcher is None:
            return
        self._watcher.stop()
        self._watcher.wait(5000)
        self._watcher = None
        self._watch_btn.setText("Watch")
        self._detect_btn.setEnabled(True)
        self._select_region_btn.setEnabled(True)
        self._set_watch_label(False)

    def _set_watch_label(self, watching: bool) -> None:
        region = self._selected_region
        if region is None:
            return
        text = f"Region: ({region.x}, {region.y}) {region.width}x{region.height}"
        if watching:
            text += "  — watching"
        self._region_label.setText(text)

    @Slot(list, str)
    def _on_watched_position(self, squares: list, fen: str) -> None:
        if self._last_fen is not None and positions_equal(fen, self._last_fen):
            # Pixels moved (cursor, highlight, animation) but the position is
            # the one already analyzed: do not run Stockfish again.
            return

        self._analysis_display.setPlainText(
            f"Position changed.\n\n{format_piece_map(squares)}\n\nFEN:\n{fen}"
        )
        self._analyze_position(fen)

    @Slot(str)
    def _on_watch_failure(self, message: str) -> None:
        self._analysis_display.setPlainText(message)

    @Slot()
    def _on_fen_submitted(self) -> None:
        fen = self._fen_input.text().strip()
        if not fen:
            return
        # Hand-typed FENs go through the same Phase 7 gate as recognized ones.
        self._analyze_position(fen)

    @Slot()
    def _on_stop_clicked(self) -> None:
        self._worker.stop()

    @Slot()
    def _on_analysis_started(self) -> None:
        self._stop_btn.setEnabled(True)
        self._depth_label.setText("Depth: analyzing...")
        self._time_label.setText("Time: analyzing...")

    @Slot(list)
    def _on_analysis_finished(self, results: list[AnalysisResult]) -> None:
        self._stop_btn.setEnabled(False)
        self._display_results(results)

    @Slot(str)
    def _on_analysis_error(self, msg: str) -> None:
        self._stop_btn.setEnabled(False)
        self._analysis_display.setPlainText(f"Analysis error:\n{msg}")
        self._on_status_changed("Error")

    @Slot(str)
    def _on_status_changed(self, status: str) -> None:
        self._engine_status = status
        current = self._status_label.text()
        engine_name = ""
        for line in current.split("\n"):
            if line.startswith("Engine:"):
                engine_name = line
                break
        self._status_label.setText(f"Status: {status}\n{engine_name}")

    def _display_results(self, results: list[AnalysisResult]) -> None:
        """Render the engine output first, the explanation after it.

        Phase 9: Stockfish is never presented as the source of a human
        explanation — the block appended here is computed from the FEN alone
        and says so in its own disclaimer.
        """
        text = format_analysis(
            results,
            engine_status=self._engine_status,
            engine_name=self._engine_name,
        )

        if results and self._last_fen:
            best_uci = results[0].pv_uci[0] if results[0].pv_uci else None
            explanation = explain_fen(self._last_fen, best_move_uci=best_uci)
            if explanation:
                text = f"{text}\n\n{explanation}"

        self._analysis_display.setPlainText(text)

        if not results:
            return

        r0 = results[0]
        self._depth_label.setText(f"Depth: {r0.depth}")
        self._time_label.setText(f"Time: {r0.time_seconds:.1f}s")

    def closeEvent(self, event) -> None:
        self._stop_watch()
        self._worker.shutdown_engine()
        event.accept()
