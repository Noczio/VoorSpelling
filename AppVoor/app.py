import sys

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QWidget, QLineEdit

from pop_up import PopUp, InfoPopUp, WarningPopUp, CriticalPopUp
from jsonInfo.welcome import WelcomeMessenger
from jsonInfo.help import HelpMessage
from load_data import LoaderCreator
from model_creation import SBSModelCreator
from estimator_creation import EstimatorCreator
from feature_selection import FeatureSelectorCreator
from parameter_search import ParameterSearchCreator
from auto_ml import JarAutoML, AutoExecutioner
from split_data import SplitterReturner
from global_vars import GlobalVariables
import forms.resources


class QDragAndDropButton(QPushButton):

    def __init__(self, parent):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._file_path = ""
        self._file_is_dropped = False

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.file_path = file_path
            self.file_is_dropped = True
            event.accept()
        else:
            self.file_is_dropped = False
            event.ignore()

    @property
    def file_path(self):
        return self._file_path

    @file_path.setter
    def file_path(self, value: str):
        self._file_path = value

    @property
    def file_is_dropped(self):
        return self._file_is_dropped

    @file_is_dropped.setter
    def file_is_dropped(self, value: bool):
        self._file_is_dropped = value


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
        self._file_type = "CSV"
        self._file_path = ""

        self.btn_back.clicked.connect(self.back)
        self.btn_info_data_type.clicked.connect(lambda: self.useful_info_pop_up("file_separation"))
        # by default is CSV, so tsv button should not be visible
        self.btn_tsv.hide()
        # change selected type to the other when clicked
        self.btn_csv.clicked.connect(lambda: self.selected_type("TSV"))
        self.btn_tsv.clicked.connect(lambda: self.selected_type("CSV"))

        self.btn_drag_file = QDragAndDropButton(self.main_area)
        self.btn_load_file = QPushButton(self.main_area)
        self.initialize_load_and_drag_file_buttons()

        self.btn_load_file.clicked.connect(self.load_dataset)
        self.btn_next.clicked.connect(self.next)

    def initialize_load_and_drag_file_buttons(self):
        self.btn_drag_file.setObjectName(u"btn_drag_file")
        self.btn_drag_file.setGeometry(QRect(360, 350, 331, 191))
        self.btn_drag_file.setStyleSheet(u"image: url(:/file/bx-file 1.svg);\n"
                                         "padding-top: 20px;\n"
                                         "padding-bottom: 80px;\n"
                                         "border-color: rgb(220, 220, 220);\n"
                                         "")
        self.btn_drag_file.setText("")
        self.btn_drag_file.raise_()

        self.btn_load_file.setObjectName(u"btn_load_file")
        self.btn_load_file.setGeometry(QRect(410, 490, 230, 40))
        font = QFont()
        font.setPointSize(14)
        self.btn_load_file.setFont(font)
        self.btn_load_file.setStyleSheet(u"QPushButton:hover{\n"
                                         "background-color: rgb(235, 225, 240);\n"
                                         "}\n"
                                         "QPushButton:pressed{\n"
                                         "background-color: rgb(220, 211, 230);\n"
                                         "border-left-color: rgb(190, 185, 220);\n"
                                         "border-top-color: rgb(190, 185, 220);\n"
                                         "border-bottom-color: rgb(215, 200, 239);\n"
                                         "border-right-color: rgb(215, 200, 239);\n"
                                         "}")
        self.btn_load_file.setText("Buscar archivo")
        self.btn_load_file.raise_()

    def selected_type(self, event):
        self._file_type = event

    def next(self) -> None:
        pop_up: PopUp = CriticalPopUp()

        if self.btn_drag_file.file_is_dropped:
            self._file_path = self.btn_drag_file.file_path

        try:
            loader = loader_creator.create_loader(self._file_path, self._file_type)
            file = loader.get_file_transformed()
            global_var.data_frame = file
            next_form = MLTypeWindow(ui_window["model"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        except FileNotFoundError:
            body = "No se encontró o no se ha subido el archivo de datos para realizar el entrenamiento"
            pop_up.open_pop_up("Error", body, "")
        except TypeError:
            body = "El archivo suministrado no cumple con los requerimientos"
            additional = "El archivo debe tener más de cien muestras y por lo menos una característica y la " \
                         "predicción. Además, es importante seleccionar correctamente si el archivo está " \
                         "separado por coma (CSV) o tabulación (TSV)"
            pop_up.open_pop_up("Error", body, additional)
        except Exception as e:
            pop_up.open_pop_up("Error", "Error desconocido", "Información detallada:" + " " + str(e))

    def back(self) -> None:
        last_form = HomeWindow(ui_window["home"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def load_dataset(self):
        file_name = QFileDialog.getOpenFileName(self, 'Seleccionar archivo', 'C:',
                                                'Text files (*.txt *.csv *.tsv)')
        self._file_path = file_name[0]


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

        self.btn_info_KNN.clicked.connect(lambda: self.useful_info_pop_up("knn"))
        self.btn_info_LinearSVC.clicked.connect(lambda: self.useful_info_pop_up("linear_svc"))
        self.btn_info_SVC_rbf.clicked.connect(lambda: self.useful_info_pop_up("svc_rbf"))
        self.btn_info_Gaussian_Naive_Bayes.clicked.connect(lambda: self.useful_info_pop_up("gaussian_naive_bayes"))

        self.btn_LinearSVC.clicked.connect(lambda: self.next("LSVC"))
        self.btn_SVC_rbf.clicked.connect(lambda: self.next("SVC"))
        self.btn_KNN.clicked.connect(lambda: self.next("KNN"))
        self.btn_Gaussian_Naive_Bayes.clicked.connect(lambda: self.next("GNB"))

    def next(self, event):
        clf = estimator_creator.create_estimator(event)
        global_var.estimator = clf
        next_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class RegressionSelection(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Lasso.clicked.connect(lambda: self.useful_info_pop_up("lasso"))
        self.btn_info_SVR_Linear.clicked.connect(lambda: self.useful_info_pop_up("linear_svr"))
        self.btn_info_SVR_rbf.clicked.connect(lambda: self.useful_info_pop_up("svr_rbf"))
        self.btn_info_SGD.clicked.connect(lambda: self.useful_info_pop_up("sgd"))

        self.btn_Lasso.clicked.connect(lambda: self.next("LASSO"))
        self.btn_SVR_Linear.clicked.connect(lambda: self.next("LSVR"))
        self.btn_SVR_rbf.clicked.connect(lambda: self.next("SVR"))
        self.btn_SGD.clicked.connect(lambda: self.next("SGD"))

    def next(self, event):
        clf = estimator_creator.create_estimator(event)
        global_var.estimator = clf
        next_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class ClusteringSelection(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Affinity_Propagation.clicked.connect(lambda: self.useful_info_pop_up("affinity_propagation"))
        self.btn_info_Minibatch_Kmeans.clicked.connect(lambda: self.useful_info_pop_up("minibatch_kmeans"))
        self.btn_info_Meanshift.clicked.connect(lambda: self.useful_info_pop_up("meanshift"))
        self.btn_info_Kmeans.clicked.connect(lambda: self.useful_info_pop_up("kmeans"))

        self.btn_Affinity_Propagation.clicked.connect(lambda: self.next("APROPAGATION"))
        self.btn_Minibatch_KMeans.clicked.connect(lambda: self.next("MINIKMEANS"))
        self.btn_Meanshift.clicked.connect(lambda: self.next("MEANSHIFT"))
        self.btn_KMeans.clicked.connect(lambda: self.next("KMEANS"))

    def next(self, event):
        clf = estimator_creator.create_estimator(event)
        global_var.estimator = clf
        next_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class WantFeatureSelection(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_FSM.clicked.connect(lambda: self.useful_info_pop_up("feature_selection"))
        self.btn_info_No_FSM.clicked.connect(lambda: self.useful_info_pop_up("no_feature_selection"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        if self.rbtn_FSM.isChecked():
            global_var.uses_feature_selection = True
            next_form = FeatureSelectionMethod(ui_window["feature_selection_method"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        else:
            global_var.uses_feature_selection = False
            next_form = WantHiperparameterSearch(ui_window["hiperparameter_search"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        last_form = MLTypeWindow(ui_window["model"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class FeatureSelectionMethod(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_FS.clicked.connect(lambda: self.useful_info_pop_up("forward_feature_selection"))
        self.btn_info_BFS.clicked.connect(lambda: self.useful_info_pop_up("backwards_feature_selection"))

        self.btn_FS.clicked.connect(lambda: self.next("FFS"))
        self.btn_BFS.clicked.connect(lambda: self.next("BFS"))

    def next(self, event):
        feature_selection_method = feature_selection_creator.create_feature_selector(event)
        global_var.feature_selection_method = feature_selection_method
        next_form = WantHiperparameterSearch(ui_window["hiperparameter_search"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self):
        last_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class WantHiperparameterSearch(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Search_Hiperparameters.clicked.connect(lambda: self.useful_info_pop_up("parameter_search"))
        self.btn_info_Hiperparameters_By_Hand.clicked.connect(lambda:
                                                              self.useful_info_pop_up("manually_set_parameters"))

        self.btn_next.clicked.connect(self.next)

    def next(self):
        if self.rbtn_Search_Hiperparameters.isChecked():
            global_var.uses_parameter_search = True
            next_form = HiperparameterMethod(ui_window["hiperparameter_search_method"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        else:
            global_var.uses_parameter_search = False
            # to do form implementation depending on estimator and if user wants or not hiperparameter search

    def back(self):
        last_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class HiperparameterMethod(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Bayesian_Search.clicked.connect(lambda: self.useful_info_pop_up("bayesian_search"))
        self.btn_info_Grid_Search.clicked.connect(lambda: self.useful_info_pop_up("grid_search"))

    def back(self):
        last_form = WantHiperparameterSearch(ui_window["hiperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


if __name__ == "__main__":

    # global_var instance to store program important variables across all forms
    global_var = GlobalVariables.get_instance()
    # create a var for each singleton creator
    loader_creator = LoaderCreator.get_instance()
    model_creator = SBSModelCreator.get_instance()
    estimator_creator = EstimatorCreator.get_instance()
    feature_selection_creator = FeatureSelectorCreator.get_instance()
    parameter_selection_creator = ParameterSearchCreator.get_instance()

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
