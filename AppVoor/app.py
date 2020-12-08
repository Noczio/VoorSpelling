import sys

from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow

from jsonInfo.welcome import WelcomeMessenger

import forms.resources

from abc import ABC, abstractmethod


class Window(QMainWindow):
    def __init__(self, window: str) -> None:
        super().__init__()
        uic.loadUi(window, self)

    @abstractmethod
    def next(self) -> None:
        pass

    @abstractmethod
    def back(self) -> None:
        pass


class HomeWindow(Window):
    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.change_message()
        self.center_window()
        self.btn_new_model.clicked.connect(self.next)

    def change_message(self) -> None:
        messenger = WelcomeMessenger(file_path=".\\jsonInfo\\welcomeMessage.json")
        text = str(messenger)
        self.lbl_phrase.setText(text)

    def center_window(self):
        frame_gm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_gm.moveCenter(center_point)
        # widget.move(QApplication.desktop().screen().rect().center() - self.rect().center())
        widget.move(frame_gm.topLeft())

    def next(self) -> None:
        widget.setCurrentIndex(widget.currentIndex()+1)

    def back(self) -> None:
        pass


class DataSetWindow(Window):
    def __init__(self, window: str):
        super().__init__(window)

    def next(self) -> None:
        pass

    def back(self) -> None:
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()

    ui_window = {"home": ".\\forms\\QT_Voorspelling_Home.ui", "dataset": ".\\forms\\QT_Voorspelling_DataSet.ui",
                 "model": ".\\forms\\QT_Voorspelling_Modelo.ui"}

    home_window = HomeWindow(ui_window["home"])
    load_dataset = DataSetWindow(ui_window["dataset"])

    gui = (home_window, load_dataset)

    for i in gui:
        widget.addWidget(i)

    widget.show()
    sys.exit(app.exec_())
