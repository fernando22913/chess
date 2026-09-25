from __future__ import annotations

from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QColor, QPainter, QPen, QGuiApplication
from PySide6.QtWidgets import QWidget

from app.capture.screen_capture import ScreenRegion


class RegionSelector(QWidget):
    """Fullscreen overlay for interactive screen region selection."""

    region_selected = Signal(ScreenRegion)
    selection_cancelled = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self._start: QPoint | None = None
        self._current: QPoint | None = None
        self._selection_rect = QRect()

    def start_selection(self) -> None:
        """Show fullscreen overlay for region selection."""
        screen = QGuiApplication.primaryScreen()
        if screen:
            geo = screen.geometry()
            self.setGeometry(geo)
        self.showFullScreen()
        self.activateWindow()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Semi-transparent dark overlay
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if not self._selection_rect.isNull():
            # Clear the selected area
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(self._selection_rect, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

            # Draw border around selection
            pen = QPen(QColor(0, 120, 215), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self._selection_rect)

            # Draw size text
            if self._start and self._current:
                w = abs(self._current.x() - self._start.x())
                h = abs(self._current.y() - self._start.y())
                text = f"{w} x {h}"
                text_x = self._selection_rect.left() + 4
                text_y = self._selection_rect.top() - 6
                if text_y < 20:
                    text_y = self._selection_rect.bottom() + 18
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_x, text_y, text)

        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._start = event.pos()
            self._current = event.pos()
            self._selection_rect = QRect(self._start, self._current).normalized()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._start is not None:
            self._current = event.pos()
            self._selection_rect = QRect(self._start, self._current).normalized()
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._start is not None:
            self._current = event.pos()
            self._selection_rect = QRect(self._start, self._current).normalized()
            self.close()

            if self._selection_rect.width() >= 5 and self._selection_rect.height() >= 5:
                region = ScreenRegion(
                    x=self._selection_rect.x(),
                    y=self._selection_rect.y(),
                    width=self._selection_rect.width(),
                    height=self._selection_rect.height(),
                )
                self.region_selected.emit(region)
            else:
                self.selection_cancelled.emit()

            self._start = None
            self._current = None
            self._selection_rect = QRect()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            self._start = None
            self._current = None
            self._selection_rect = QRect()
            self.selection_cancelled.emit()
