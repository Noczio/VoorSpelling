from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt5.QtWidgets import QPushButton, QFileDialog


class QDragAndDropButton(QPushButton):
    loaded = pyqtSignal()

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._file_path: str = ""

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
            self.loaded.emit()
            event.accept()
        else:
            event.ignore()

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value


class QLoadButton(QPushButton):
    loaded = pyqtSignal()

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self._file_path: str = ""
        self.clicked.connect(self.on_click)

    def on_click(self) -> None:
        file_name = QFileDialog.getOpenFileName(self, 'Seleccionar archivo', 'C:',
                                                'Text files (*.txt *.csv *.tsv)')
        self.file_path = file_name[0]
        self.loaded.emit()

    @property
    def file_path(self) -> str:
        return self._file_path

    @file_path.setter
    def file_path(self, value: str) -> None:
        self._file_path = value

