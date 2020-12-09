import sys

from PyQt5 import QtWidgets
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from abc import ABC, abstractmethod

from jsonInfo.welcome import WelcomeMessenger
from jsonInfo.help import HelpMessage
import forms.resources


class Window(QMainWindow):
    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__()
        uic.loadUi(window, self)
        self._help_message = HelpMessage(file_path=help_message_path)

    def next(self) -> None:
        pass

    def back(self) -> None:
        pass

    def show_info(self, key: str):
        title, body, example, url = self._help_message[key]
        msg = QMessageBox()
        msg.setWindowTitle(title)
        if example == "":
            msg.setText(body)
        else:
            msg.setText(body+"\n\n"+"Ejemplo:"+" "+example)
        if url is not "":
            msg.setInformativeText("\n\n"+"Para más información visitar:"+" "+url)
        msg.setIcon(QMessageBox.Information)
        msg.setStandardButtons(QMessageBox.Ok)
        _ = msg.exec_()  # this will show our messagebox


class HomeWindow(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self._change_message()
        self.btn_next.clicked.connect(self.next)

    def _change_message(self) -> None:
        messenger = WelcomeMessenger(file_path=".\\jsonInfo\\welcomeMessage.json")
        text = str(messenger)
        self.lbl_description.setText(text)

    def center_window(self):
        frame_gm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_gm.moveCenter(center_point)
        widget.move(frame_gm.topLeft())

    def next(self) -> None:
        next_form = DataSetWindow(ui_window["dataset"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class DataSetWindow(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json"):
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)
        self.btn_info_data_type.clicked.connect(self.info_data_type)

    def next(self) -> None:
        # to do. Validate if data is a valid dataset and some option has been selected
        next_form = MLTypeWindow(ui_window["model"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = HomeWindow(ui_window["home"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def info_data_type(self):
        self.show_info("file_separation")


class MLTypeWindow(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json"):
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

    def next(self) -> None:
        # to do. Validate if any of the radio buttons are selected and the go the next form
        pass

    def back(self) -> None:
        last_form = DataSetWindow(ui_window["dataset"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


if __name__ == "__main__":

    ui_window = {"home": ".\\forms\\QT_Voorspelling_Home.ui",
                 "dataset": ".\\forms\\QT_Voorspelling_DataSet.ui",
                 "model": ".\\forms\\QT_Voorspelling_Modelo.ui",
                 "prediction_type": ".\\forms\\QT_Voorspelling_TipoPrediccion.ui",
                 "classification": ".\\forms\\QT_Voorspelling_TipoP_Clasificacion.ui",
                 "regression": ".\\forms\\QT_Voorspelling_TipoP_Regresion.ui",
                 "clustering": ".\\forms\\QT_Voorspelling_TipoP_Agrupamiento.ui",
                 "feature_selection": ".\\forms\\QT_Voorspelling_Caracteristicas.ui",
                 "feature_selection_method": ".\\forms\\QT_Voorspelling_CaracteristicasMetodo.ui",
                 "hiperparameter_search": ".\\forms\\QT_Voorspelling_Hiperparametros.ui",
                 "hiperparameter_search_method": ".\\forms\\QT_Voorspelling_HiperparametrosMetodo.ui",
                 "result_screen": ".\\forms\\QT_Voorspelling_Resultado.ui"}

    app = QApplication(sys.argv)
    widget = QtWidgets.QStackedWidget()
    home_window = HomeWindow(ui_window["home"])
    home_window.center_window()
    widget.addWidget(home_window)
    widget.show()
    sys.exit(app.exec_())
