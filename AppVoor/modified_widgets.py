import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton

DateTime = datetime


class QDragAndDropButton(QPushButton):

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._file_path: str = ""
        self._time_loaded: DateTime = None

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.file_path = file_path
            self.time_loaded = datetime.datetime.now()
            event.accept()
        else:
            event.ignore()

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value

    @property
    def time_loaded(self) -> DateTime:
        return self._time_loaded

    @time_loaded.setter
    def time_loaded(self, value: DateTime) -> None:
        self._time_loaded = value
