import sys

import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRect, QThreadPool
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication

from auto_ml import JarAutoML, AutoExecutioner
from estimator_creation import EstimatorCreator
from feature_selection import FeatureSelectorCreator
from global_vars import GlobalVariables
from jsonInfo.welcome import WelcomeMessenger
from load_data import LoaderCreator
from model_creation import SBSModelCreator
from parameter_search import ParameterSearchCreator
from pop_up import PopUp, WarningPopUp, CriticalPopUp
from modified_widgets import QDragAndDropButton, QLoadButton
from parallel import Worker
from view import Window
from forms import resources

DataFrame = pd.DataFrame
NpArray = np.ndarray


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
        self.file_type: str = "CSV"
        self.last = "any"

        self.btn_drag_file = QDragAndDropButton(self.main_area)
        self.btn_drag_file.setObjectName(u"btn_drag_file")
        self.btn_drag_file.setGeometry(QRect(360, 350, 331, 191))
        self.btn_drag_file.setStyleSheet(u"image: url(:/file/bx-file 1.svg);\n"
                                         "padding-top: 20px;\n"
                                         "padding-bottom: 80px;\n"
                                         "border-color: rgb(220, 220, 220);\n"
                                         "")
        self.btn_drag_file.setText("")
        self.btn_drag_file.raise_()

        self.btn_load_file = QLoadButton(self.main_area)
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

        self.btn_back.clicked.connect(self.back)
        self.btn_info_data_type.clicked.connect(lambda: self.useful_info_pop_up("file_separation"))
        # by default is CSV, so tsv button should not be visible
        self.btn_tsv.hide()
        # change selected type to the other when clicked
        self.btn_csv.clicked.connect(lambda: self.select_file_type("TSV"))
        self.btn_tsv.clicked.connect(lambda: self.select_file_type("CSV"))

        self.btn_load_file.loaded.connect(lambda: self.set_last_emitted("load_file"))
        self.btn_drag_file.loaded.connect(lambda: self.set_last_emitted("drag_file"))
        self.btn_next.clicked.connect(self.next)

    def select_file_type(self, event):
        self.file_type = event

    def set_last_emitted(self, event):
        if event == "load_file":
            self.last = "loaded"
            self.btn_load_file.setStyleSheet(u"QPushButton{\n"
                                             "border-color: rgb(60,179,113);\n"
                                             "}\n"
                                             u"QPushButton:hover{\n"
                                             "background-color: rgb(235, 225, 240);\n"
                                             "}\n"
                                             "QPushButton:pressed{\n"
                                             "background-color: rgb(220, 211, 230);\n"
                                             "border-left-color: rgb(60,179,113);\n"
                                             "border-top-color: rgb(60,179,113);\n"
                                             "border-bottom-color: rgb(85, 194, 132);\n"
                                             "border-right-color: rgb(85, 194, 132);\n"
                                             "}")
            self.btn_drag_file.setStyleSheet(u"image: url(:/file/bx-file 1.svg);\n"
                                             "padding-top: 20px;\n"
                                             "padding-bottom: 80px;\n"
                                             "border-color: rgb(220, 220, 220);\n"
                                             "")
        else:
            self.last = "dropped"
            self.btn_drag_file.setStyleSheet(u"image: url(:/file/bx-file 1.svg);\n"
                                             "padding-top: 20px;\n"
                                             "padding-bottom: 80px;\n"
                                             "border-color: rgb(60,179,113);\n"
                                             "")
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

    def next(self) -> None:
        pop_up: PopUp = CriticalPopUp()
        if self.last == "any":
            body = "Ningun archivo ha sido seleccionado. Por favor subir un archivo y seleccionar la separación del " \
                   "mismo, ya sea TSV o CSV "
            pop_up.open_pop_up("Error", body, "")
        else:
            try:
                if self.last == "loaded":
                    loader = loader_creator.create_loader(self.btn_load_file.file_path, self.file_type)
                else:
                    loader = loader_creator.create_loader(self.btn_drag_file.file_path, self.file_type)
                file = loader.get_file_transformed()
                global_var.data_frame = file
                next_form = MLTypeWindow(ui_window["model"])
                widget.addWidget(next_form)
                widget.removeWidget(widget.currentWidget())
                widget.setCurrentIndex(widget.currentIndex())
            except FileNotFoundError:
                body = "No se encontró el archivo de datos para realizar el entrenamiento"
                pop_up.open_pop_up("Error", body, "")
            except TypeError:
                body = "El archivo suministrado no cumple con los requerimientos"
                additional = "El archivo debe tener más de cien muestras y por lo menos una característica y la " \
                             "predicción. Además, es importante seleccionar correctamente si el archivo está " \
                             "separado por coma (CSV) o tabulación (TSV)"
                pop_up.open_pop_up("Error", body, additional)
            except ValueError:
                body = "El archivo seleccionado no cumple con los requerimientos" \
                       " para ser considerado un archivo de texto con las extensiones permitidas"
                pop_up.open_pop_up("Error", body, "")
            except OSError:
                body = "El archivo seleccionado no cumple con los requerimientos" \
                       " para ser considerado un archivo de texto"
                additional = "No debe tener una extensión diferente a .txt .csv o .tsv"
                pop_up.open_pop_up("Error", body, additional)
            except Exception as e:
                body = "El archivo seleccionado no cumple con los requerimientos para ser utilizado para entrenar un " \
                       "modelo de inteligencia artificial"
                additional = "Información detallada:" + " " + str(e)
                pop_up.open_pop_up("Error", body, additional)

    def back(self) -> None:
        global_var.reset()
        last_form = HomeWindow(ui_window["home"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


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
        global_var.reset("data_set")
        last_form = DataSetWindow(ui_window["dataset"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class AutoLoad(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.thread_pool = QThreadPool()

        self.ml_worker = Worker(self.train_model)
        self.ml_worker.signals.result.connect(self.add_info)
        self.ml_worker.signals.finished.connect(self.next)

        self.thread_pool.start(self.ml_worker)

        self.lbl_cancel.mouseReleaseEvent = self.back

    def add_info(self, text: any) -> None:
        self.ted_info.append(str(text))

    def train_model(self, progress_callback):
        try:
            self.add_info("Inicio de proceso")
            automl_ml = JarAutoML(10, False, 5000)
            model = AutoExecutioner(automl_ml)
            data_frame = global_var.data_frame
            self.add_info("Datos cargados")
            model.train_model(data_frame)
        except Exception as e:
            self.add_info("Error durante el proceso")
            pop_up: PopUp = CriticalPopUp()
            body = "Ocurrió un error durante el proceso de entrenamiento automatizado con Jar AutoMl." \
                   " Será redirigido a la página de inicio"
            additional = "Por favor verificar los datos suministrados para futuros entrenamientos" + \
                         "\n\nInformación detallada:" + " " + str(e)
            pop_up.open_pop_up("Error", body, additional)
            global_var.reset()
            last_form = HomeWindow(ui_window["home"])
            widget.addWidget(last_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def next(self):
        # to do, final result form
        self.add_info("Proceso completado")

    def back(self, event):
        pop_up: PopUp = WarningPopUp()
        title = "Cancelar entrenamiento"
        body = "¿Estas seguro que deseas cancelar el entrenamiento?. Toda la información será eliminada."
        answer = pop_up.open_pop_up(title, body, "")
        if answer:
            self.thread_pool.cancel(self.ml_worker)
            global_var.reset()
            last_form = HomeWindow(ui_window["home"])
            widget.addWidget(last_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())


class StepByStepLoad(Window):

    def __init__(self, window: str, help_message_path: str = ".\\jsonInfo\\helpMessage.json") -> None:
        super().__init__(window, help_message_path)
        self.thread_pool = QThreadPool()

        self.ml_worker = Worker(self.train_model)
        self.ml_worker.signals.result.connect(self.add_info)
        self.ml_worker.signals.finished.connect(self.next)

        self.thread_pool.start(self.ml_worker)

        self.lbl_cancel.mouseReleaseEvent = self.back

    def add_info(self, text: any) -> None:
        self.ted_info.append(str(text))

    def train_model(self):
        try:
            self.add_info("Inicio de proceso")
            automl_ml = JarAutoML(10, False, 5000)
            model = AutoExecutioner(automl_ml)
            data_frame = global_var.data_frame
            self.add_info("Datos cargados")
            model.train_model(data_frame)
        except Exception as e:
            self.add_info("Error durante el proceso")
            pop_up: PopUp = CriticalPopUp()
            body = "Ocurrió un error durante el proceso de entrenamiento automatizado con Jar AutoMl." \
                   " Será redirigido a la página de inicio"
            additional = "Por favor verificar los datos suministrados para futuros entrenamientos" + \
                         "\n\nInformación detallada:" + " " + str(e)
            pop_up.open_pop_up("Error", body, additional)
            global_var.reset()
            last_form = HomeWindow(ui_window["home"])
            widget.addWidget(last_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def next(self):
        # to do, final result form
        self.add_info("Proceso completado")

    def back(self, event):
        pop_up: PopUp = WarningPopUp()
        title = "Cancelar entrenamiento"
        body = "¿Estas seguro que deseas cancelar el entrenamiento?. Toda la información será eliminada."
        answer = pop_up.open_pop_up(title, body, "")
        if answer:
            self.thread_pool.cancel(self.ml_worker)
            global_var.reset()
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
        global_var.reset(estimator=None)
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
        global_var.reset(estimator=None)
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
        global_var.reset(estimator=None)
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
        global_var.reset(estimator=None)
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
        global_var.reset("uses_feature_selection", "feature_selection_method")
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
        global_var.reset("uses_feature_selection", "uses_parameter_search")
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
        global_var.reset("uses_parameter_search", "parameter_search_method")
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
