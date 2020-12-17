from abc import ABC, abstractmethod

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMessageBox


class PopUp(ABC):

    @abstractmethod
    def open_pop_up(self, title: str, body: str, additional: str) -> bool:
        pass


class InfoPopUp(PopUp):

    def open_pop_up(self, title: str, body: str, additional: str) -> bool:
        # set font as QFont object
        font = QFont()
        font.setFamily("MS Shell")
        font.setPointSize(11)
        # create a QMessageBox object and then set its values
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(body)

        if additional is not "":
            msg.setInformativeText(additional)

        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setFont(font)

        _ = msg.exec_()  # this will show the messagebox
        return True


class WarningPopUp(PopUp):

    def open_pop_up(self, title: str, body: str, additional: str) -> bool:
        # set font as QFont object
        font = QFont()
        font.setFamily("MS Shell")
        font.setPointSize(11)
        # create a QMessageBox object and then set its values
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(body)

        if additional is not "":
            msg.setInformativeText(additional)

        msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setFont(font)

        return_value = msg.exec_()  # this will show the messagebox
        if return_value == QMessageBox.Yes:
            return True
        else:
            return False
