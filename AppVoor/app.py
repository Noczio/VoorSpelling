import sys

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow

from pop_up import PopUp, InfoPopUp, WarningPopUp
from jsonInfo.welcome import WelcomeMessenger
from jsonInfo.help import HelpMessage
from load_data import LoaderCreator
from model_creation import SBSModelCreator
from estimator_creation import EstimatorCreator
from feature_selection import FeatureSelection, FeatureSelectorCreator
from parameter_search import ParameterSearch, ParameterSearchCreator
from auto_ml import JarAutoML, AutoExecutioner
from split_data import SplitterReturner
import forms.resources

import pandas as pd
import numpy as np
from typing import Any

DataFrame = pd.DataFrame
NpArray = np.ndarray


class GlobalVariables:

    _df: DataFrame
    _fs: bool
    _ps: bool
    _fsm: FeatureSelection
    _psm: ParameterSearch
    _clf: Any
    __instance = None

    @staticmethod
    def get_instance() -> "GlobalVariables":
        """Static access method."""
        if GlobalVariables.__instance is None:
            GlobalVariables()
        return GlobalVariables.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if GlobalVariables.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            GlobalVariables.__instance = self

    @property
    def data_frame(self) -> DataFrame:
        return self._df

    @data_frame.setter
    def data_frame(self, value: DataFrame) -> None:
        self._df = value

    @property
    def uses_feature_selection(self) -> bool:
        return self._fs

    @uses_feature_selection.setter
    def uses_feature_selection(self, value: bool) -> None:
        self._fs = value

    @property
    def uses_parameter_search(self) -> bool:
        return self._ps

    @uses_parameter_search.setter
    def uses_parameter_search(self, value: bool) -> None:
        self._ps = value

    @property
    def feature_selection_method(self) -> FeatureSelection:
        return self._fsm

    @feature_selection_method.setter
    def feature_selection_method(self, value: FeatureSelection) -> None:
        self._fsm = value

    @property
    def parameter_search_method(self) -> ParameterSearch:
        return self._psm

    @parameter_search_method.setter
    def parameter_search_method(self, value: ParameterSearch) -> None:
        self._psm = value

    @property
    def estimator(self) -> Any:
        return self._clf

    @estimator.setter
    def estimator(self, value: Any) -> None:
        self._clf = value

class Window(QMainWindow):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__()
        uic.loadUi(window, self)
        self._help_message = HelpMessage(file_path=help_message_path)

    def useful_info_pop_up(self, key: str) -> None:
        # get help message info from HelpMessage object
        title, body, example, url = self._help_message[key]
        if example is not "":
            body = body + "\n\n" + "Ejemplo:" + "\n\n" + example
        if url is not "":
            url = "Para más información vistiar:" + " " + url
        # call general_info_pop_up with useful_info_pop_up info
        pop_up: PopUp = InfoPopUp()
        pop_up.open_pop_up(title, body, url)

    def next(self, *args, **kwargs) -> None:
        pass

    def back(self, *args, **kwargs) -> None:
        pass


class HomeWindow(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self._change_message()
        self.btn_next.clicked.connect(self.next)

    def _change_message(self) -> None:
        messenger = WelcomeMessenger(file_path=".\\jsonInfo\\welcomeMessage.json")
        text = str(messenger)
        self.lbl_description.setText(text)

    def center_window(self) -> None:
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

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)
        self.btn_next.clicked.connect(self.next)
        self.btn_info_data_type.clicked.connect(lambda: self.useful_info_pop_up("file_separation"))
        self._file_type = "CSV"
        self._selected_data_path = ""

    def next(self) -> None:
        # to do. Validate if info is correct. If it is not don't continue and show pop up
        # loader_creator = LoaderCreator.get_instance()
        # loader = loader_creator.create_loader(selected_data, self._file_type)
        # file_transformed = loader.get_file_transformed()
        next_form = MLTypeWindow(ui_window["model"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = HomeWindow(ui_window["home"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def load_dataset(self):
        # to do. Open a file browser a store path into _selected_data_path
        pass


class MLTypeWindow(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)
        self.btn_info_automl.clicked.connect(lambda: self.useful_info_pop_up("auto_machine_learning"))
        self.btn_info_sbsml.clicked.connect(lambda: self.useful_info_pop_up("step_by_step"))
        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        if self.rbtn_sbsml.isChecked():
            next_form = PredictionType(ui_window["prediction_type"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        else:
            next_form = AutoLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = DataSetWindow(ui_window["dataset"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class AutoLoad(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.lbl_cancel.mouseReleaseEvent = self.back

    def back(self, event):
        pop_up: PopUp = WarningPopUp()
        title = "Cancelar entrenamiento"
        body = "¿Estas seguro que deseas cancelar el entrenamiento?. Toda la información será eliminada."
        answer = pop_up.open_pop_up(title, body, "")
        if answer:
            last_form = HomeWindow(ui_window["home"])
            widget.addWidget(last_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())


class StepByStepLoad(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.lbl_cancel.mouseReleaseEvent = self.back

    def back(self, event):
        pop_up: PopUp = WarningPopUp()
        title = "Cancelar entrenamiento"
        body = "¿Estas seguro que deseas cancelar el entrenamiento?. Toda la información será eliminada."
        answer = pop_up.open_pop_up(title, body, "")
        if answer:
            last_form = HomeWindow(ui_window["home"])
            widget.addWidget(last_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())


class PredictionType(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)
        self.btn_info_Classification.clicked.connect(lambda: self.useful_info_pop_up("classification"))
        self.btn_info_Regression.clicked.connect(lambda: self.useful_info_pop_up("regression"))
        self.btn_info_Clustering.clicked.connect(lambda: self.useful_info_pop_up("clustering"))
        self.btn_Classification.clicked.connect(lambda: self.next("classification"))
        self.btn_Regression.clicked.connect(lambda: self.next("regression"))
        self.btn_Clustering.clicked.connect(lambda: self.next("clustering"))

    def next(self, event) -> None:
        possibilities = {"classification": ClassificationSelection(ui_window["classification"]),
                         "regression": RegressionSelection(ui_window["regression"]),
                         "clustering": ClusteringSelection(ui_window["clustering"])}

        next_form = possibilities[event]
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = MLTypeWindow(ui_window["model"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class ClassificationSelection(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

    def back(self) -> None:
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class RegressionSelection(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

    def back(self) -> None:
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class ClusteringSelection(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

    def back(self) -> None:
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


if __name__ == "__main__":

    # global_var instance to store program important variables across all forms
    global_var = GlobalVariables.get_instance()

    # dict with path to every view
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
    # by default first form is home
    home_window = HomeWindow(ui_window["home"])
    home_window.center_window()
    # set first view in widgetStack, its min and max size. Finally, show it and start app logic
    widget.addWidget(home_window)
    widget.setMaximumSize(1440, 1024)
    widget.setMinimumSize(1440, 1024)
    widget.show()
    sys.exit(app.exec_())
