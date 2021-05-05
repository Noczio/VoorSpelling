import sys
from typing import Any

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRect, QThreadPool, QThread, QSize
from PyQt5.QtGui import QFont, QTextCursor, QIcon
from PyQt5.QtWidgets import QApplication

from resources.backend_scripts.auto_ml import JarAutoML, AutoExecutioner
from resources.backend_scripts.estimator_creation import EstimatorCreator
from resources.backend_scripts.feature_selection import FeatureSelectorCreator, FeatureSelection
from resources.backend_scripts.global_vars import GlobalVariables
from resources.backend_scripts.load_data import LoaderCreator
from resources.backend_scripts.model_creation import SBSModelCreator
from resources.backend_scripts.parameter_search import BayesianSearchParametersPossibilities
from resources.backend_scripts.parameter_search import GridSearchParametersPossibilities
from resources.backend_scripts.parameter_search import ParameterSearchCreator, ParameterSearch
from resources.backend_scripts.result_creation import FCreator, SBSResult
from resources.backend_scripts.switcher import Switch
from resources.forms import QT_resources
from resources.frontend_scripts.modified_widgets import QDragAndDropButton, QLoadButton
from resources.frontend_scripts.parallel import LongWorker, EmittingStream
from resources.frontend_scripts.pop_up import PopUp, WarningPopUp, CriticalPopUp
from resources.frontend_scripts.view import Window
from resources.json_info.welcome import WelcomeMessenger
from resources.ui_path import ui_window, ui_icons

DataFrame = pd.DataFrame
NpArray = np.ndarray


class PredictionTypePossibilities(Switch):

    @staticmethod
    def classification() -> Window:
        return ClassificationSelection(ui_window["classification"])

    @staticmethod
    def regression() -> Window:
        return RegressionSelection(ui_window["regression"])

    @staticmethod
    def clustering() -> Window:
        return ClusteringSelection(ui_window["clustering"])


class EstimatorParametersPossibilities(Switch):

    @staticmethod
    def LinearSVC() -> Window:
        return LinearSVCParameters(ui_window["LinearSVC"])

    @staticmethod
    def SVC() -> Window:
        return SVCParameters(ui_window["SVC"])

    @staticmethod
    def KNeighborsClassifier() -> Window:
        return KNeighborsClassifierParameters(ui_window["KNeighborsClassifier"])

    @staticmethod
    def GaussianNB() -> Window:
        return GaussianNBParameters(ui_window["GaussianNB"])

    @staticmethod
    def LinearSVR() -> Window:
        return LinearSVRParameters(ui_window["LinearSVR"])

    @staticmethod
    def SVR() -> Window:
        return SVRParameters(ui_window["SVR"])

    @staticmethod
    def Lasso() -> Window:
        return LassoParameters(ui_window["Lasso"])

    @staticmethod
    def SGDClassifier() -> Window:
        return SGDClassifierParameters(ui_window["SGDClassifier"])

    @staticmethod
    def AffinityPropagation() -> Window:
        return AffinityPropagationParameters(ui_window["AffinityPropagation"])

    @staticmethod
    def KMeans() -> Window:
        return KMeansParameters(ui_window["KMeans"])

    @staticmethod
    def MiniBatchKMeans() -> Window:
        return MiniBatchKMeansParameters(ui_window["MiniBatchKMeans"])

    @staticmethod
    def MeanShift() -> Window:
        return MeanShiftParameters(ui_window["MeanShift"])


class HomeWindow(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_next.clicked.connect(self.next)

    def on_load(self) -> None:
        super(HomeWindow, self).on_load()
        messenger = WelcomeMessenger(".\\resources\\json_info\\welcome_message.json")
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

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.file_type: str = "CSV"
        self.last: str = "any"
        self.critical_pop_up: PopUp = CriticalPopUp()

        self.btn_drag_file = QDragAndDropButton(self.main_area)
        self.btn_load_file = QLoadButton(self.main_area)
        self.set_load_and_drag_buttons()

        self.btn_back.clicked.connect(self.back)
        self.btn_info_data_type.clicked.connect(lambda: self.useful_info_pop_up("file_separation"))

        # change selected type to the other when clicked
        self.btn_csv.clicked.connect(lambda: self.select_file_type("TSV"))
        self.btn_tsv.clicked.connect(lambda: self.select_file_type("CSV"))

        self.btn_load_file.loaded.connect(lambda: self.set_last_emitted("load_file"))
        self.btn_drag_file.loaded.connect(lambda: self.set_last_emitted("drag_file"))
        self.btn_next.clicked.connect(self.handle_file)

    def set_load_and_drag_buttons(self) -> None:
        # by default is CSV, so tsv button should not be visible
        self.btn_tsv.hide()
        # file buttons set geometry and style
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

    def select_file_type(self, event) -> None:
        self.file_type = event

    def set_last_emitted(self, event) -> None:
        if event == "load_file" and self.btn_load_file.file_path is not "":
            self.last = "load_file"
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
        elif event == "drag_file" and self.btn_drag_file.file_path is not "":
            self.last = "drag_file"
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
        next_form = MLTypeWindow(ui_window["model"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def handle_error(self, error) -> None:
        body, additional = "", ""
        if type(error) == FileNotFoundError:
            body = "No se encontró el archivo de datos para realizar el entrenamiento"
        elif type(error) == TypeError:
            body = "El archivo suministrado no cumple con los requerimientos"
            additional = "El archivo debe tener más de cien muestras y por lo menos una característica y la " \
                         "predicción. Además, es importante seleccionar correctamente si el archivo está " \
                         "separado por coma (CSV) o tabulación (TSV)"
        elif type(error) == ValueError:
            body = "El archivo seleccionado no cumple con los requerimientos" \
                   " para ser considerado un archivo de texto con las extensiones permitidas"
        elif type(error) == OSError:
            body = "El archivo seleccionado no cumple con los requerimientos" \
                   " para ser considerado un archivo de texto"
            additional = "No debe tener una extensión diferente a .txt .csv o .tsv"
        else:
            body = "El archivo seleccionado no cumple con los requerimientos para ser utilizado para entrenar un " \
                   "modelo de inteligencia artificial"
            additional = "Información detallada:" + " " + str(error)
        self.critical_pop_up.open_pop_up("Error", body, additional)

    def handle_file(self) -> None:
        all_ok: bool = False
        try:
            if self.last == "any":
                body = "Ningún archivo ha sido seleccionado. Por favor subir un archivo " \
                       "y seleccionar la separación del mismo, ya sea TSV o CSV "
                self.critical_pop_up.open_pop_up("Error", body, "")
            else:
                loader_creator = LoaderCreator.get_instance()
                if self.last == "load_file":
                    loader = loader_creator.create_loader(self.btn_load_file.file_path, self.file_type)
                else:
                    loader = loader_creator.create_loader(self.btn_drag_file.file_path, self.file_type)
                file = loader.get_file_transformed()
                global_var.data_frame = file
                all_ok = True
        except Exception as e:
            self.handle_error(e)
            all_ok = False
        finally:
            if all_ok:
                self.next()

    def back(self) -> None:
        global_var.reset("data_set")
        last_form = HomeWindow(ui_window["home"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class MLTypeWindow(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
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
            result = self.last_warning_pop_up()
            if result:
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

    def __init__(self, window: str) -> None:
        super().__init__(window)
        sys.stdout = EmittingStream(textWritten=self.add_info)

        self.lbl_cancel.mouseReleaseEvent = self.cancel_training
        # when form opens, create a thread pool and a worker
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)
        self.ml_worker = LongWorker()
        self.ml_worker.set_params(self.train_model)
        self.ml_worker.signals.program_finished.connect(self.next)
        self.ml_worker.signals.program_error.connect(self.handle_error)
        self.ml_worker.signals.result.connect(self.add_info)
        self.ml_worker.setAutoDelete(True)
        self.thread_pool.start(self.ml_worker, priority=1)

    def close_window(self) -> None:
        super(AutoLoad, self).close_window()
        widget.close()

    def next(self) -> None:
        QThread.sleep(1)
        next_form = FinalResult(ui_window["result_final"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def train_model(self) -> None:
        automl_ml = JarAutoML(10, False, 5000)
        model = AutoExecutioner(automl_ml)
        print(str(model) + "\n\n")
        data_frame = global_var.data_frame
        model.train_model(data_frame)
        print("Process finished successfully")

    def last_warning_pop_up(self) -> bool:
        pop_up: PopUp = WarningPopUp()
        title = "Cancelar entrenamiento"
        body = "¿Estas seguro que deseas cancelar el entrenamiento?"
        additional = "La aplicación se cerrará para evitar conflictos con las variables utilizadas hasta el momento."
        answer = pop_up.open_pop_up(title, body, additional)
        return answer

    def cancel_training(self, event) -> None:
        """Show a Warning pop up and then if user wants to finished the app, close it"""
        event.accept()
        result = self.last_warning_pop_up()
        if result:
            self.thread_pool.cancel(self.ml_worker)
            self.close_window()

    def handle_error(self, error) -> None:
        """Print error message to the QTextEdit"""

        def write_error():
            for i in info:
                self.add_info(i)
                QThread.sleep(1)

        # deactivate lbl press behaviour due to an error
        self.lbl_cancel.mouseReleaseEvent = None
        self.lbl_cancel.setStyleSheet(u"QLabel\n"
                                      "{\n"
                                      "   border: none;\n"
                                      "	color: rgb(105, 105, 105);\n"
                                      "}")
        info = ("Error\n", str(error), "\nCerrando aplicación para evitar conflictos de memoria" + " ", ".", ".", ".")
        # worker to write info to ted_info
        temp_worker = LongWorker(func=write_error)
        temp_worker.signals.program_finished.connect(self.close_window)
        self.thread_pool.start(temp_worker, priority=2)

    def add_info(self, info: Any) -> None:
        """Append text to the QTextEdit."""
        message: str = ""
        try:
            message = str(info)
        except Exception as e:
            message = str(e)
        finally:
            cursor = self.ted_info.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(message)
            self.ted_info.setTextCursor(cursor)
            self.ted_info.ensureCursorVisible()


class StepByStepLoad(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        sys.stdout = EmittingStream(textWritten=self.add_info)

        self.lbl_cancel.mouseReleaseEvent = self.cancel_training
        # when form opens, create a thread pool and a worker
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(2)
        self.ml_worker = LongWorker()
        self.ml_worker.set_params(self.train_model)
        self.ml_worker.signals.program_finished.connect(self.next)
        self.ml_worker.signals.program_error.connect(self.handle_error)
        self.ml_worker.signals.result.connect(self.add_info)
        self.ml_worker.setAutoDelete(True)
        self.thread_pool.start(self.ml_worker, priority=1)

    def close_window(self) -> None:
        super(StepByStepLoad, self).close_window()
        widget.close()

    def next(self) -> None:
        QThread.sleep(1)
        next_form = FinalResult(ui_window["result_final"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def save_results(self, score_text: str, estimator: Any, initial_parameters: dict, best_features: list,
                     best_parameters: dict, feature_selector: FeatureSelection, parameter_selector: ParameterSearch,
                     folder_path: str) -> None:
        # App is set up to be used by spanish speakers, so prediction type must be translated for further use
        translation = {"classification": "clasificación",
                       "regression": "regresión",
                       "clustering": "agrupamiento"}
        prediction_type_text = f"{translation[global_var.prediction_type]} ({global_var.prediction_type}) paso a paso"
        # All important info is storage in a variable to be displayed in a markdown file as a 2x5 table
        info = ["Opción", "Selección",
                "Tipo de predicción", prediction_type_text,
                "Estimador", estimator.__class__.__name__,
                "Selección de características", "No" if feature_selector is None
                else feature_selector.__class__.__name__,
                "Selección de hiperparámetros", "No" if parameter_selector is None
                else parameter_selector.__class__.__name__
                ]
        table = {"columns": 2, "rows": 5, "info": info}
        print("Saving results document ...")
        # save estimator info results into markdown file
        SBSResult.estimator_info(table, best_features, initial_parameters, best_parameters, score_text, folder_path)
        print("Saving estimator ...")
        SBSResult.dump_estimator(estimator, folder_path)

    def save_logs(self, folder_path: str) -> None:
        print("Saving console logs ...")
        # Finally, after all is finished write ted info to its markdown file
        ted_text = self.ted_info.toPlainText()
        fixed_ted_text = ted_text.split("\n")
        SBSResult.console_info(fixed_ted_text, folder_path)

    def train_model(self) -> None:
        # gets important info and the scores model
        model_creator = SBSModelCreator.get_instance()
        model = model_creator.create_model(global_var.uses_feature_selection, global_var.uses_parameter_search)
        model.estimator = global_var.estimator
        model.initial_parameters = global_var.parameters
        model.feature_selector = global_var.feature_selection_method
        model.parameter_selector = global_var.parameter_search_method
        # scoring metric by default for each type of prediction
        score_type = {"classification": "accuracy",
                      "regression": "neg_mean_squared_error",
                      "clustering": "mutual_info_score"}
        print("Training ...")
        # score model, then get a user friendly message for that score and finally return data
        score = model.score_model(global_var.data_frame, score_type[global_var.prediction_type], 10)
        score_text = f"Rendimiento promedio \"{score_type[global_var.prediction_type]}\": {score}"
        f_creator = FCreator()
        folder_path = f_creator.folder_path
        self.save_results(score_text, model.estimator, model.initial_parameters, list(model.best_features),
                          model.best_parameters, model.feature_selector, model.parameter_selector, folder_path)
        self.save_logs(folder_path)
        print("Process finished successfully")

    def last_warning_pop_up(self) -> bool:
        pop_up: PopUp = WarningPopUp()
        title = "Cancelar entrenamiento"
        body = "¿Estas seguro que deseas cancelar el entrenamiento?"
        additional = "La aplicación se cerrará para evitar conflictos con las variables utilizadas hasta el momento."
        answer = pop_up.open_pop_up(title, body, additional)
        return answer

    def cancel_training(self, event) -> None:
        """Show a Warning pop up and then if user wants to finished the app, close it"""
        event.accept()
        result = self.last_warning_pop_up()
        if result:
            self.thread_pool.cancel(self.ml_worker)
            self.close_window()

    def handle_error(self, error) -> None:
        """Print error message to the QTextEdit"""

        def write_error():
            for i in info:
                self.add_info(i)
                QThread.sleep(1)

        # deactivate lbl press behaviour due to an error
        self.lbl_cancel.mouseReleaseEvent = None
        self.lbl_cancel.setStyleSheet(u"QLabel\n"
                                      "{\n"
                                      "   border: none;\n"
                                      "	color: rgb(105, 105, 105);\n"
                                      "}")
        info = ("Error\n", str(error), "\nCerrando aplicación para evitar conflictos de memoria" + " ", ".", ".", ".")
        # worker to write info to ted_info
        temp_worker = LongWorker(func=write_error)
        temp_worker.signals.program_finished.connect(self.close_window)
        self.thread_pool.start(temp_worker, priority=2)

    def add_info(self, info: Any) -> None:
        """Append text to the QTextEdit."""
        message: str = ""
        try:
            message = str(info)
        except Exception as e:
            message = str(e)
        finally:
            cursor = self.ted_info.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(message)
            self.ted_info.setTextCursor(cursor)
            self.ted_info.ensureCursorVisible()


class PredictionType(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Classification.clicked.connect(lambda: self.useful_info_pop_up("classification"))
        self.btn_info_Regression.clicked.connect(lambda: self.useful_info_pop_up("regression"))
        self.btn_info_Clustering.clicked.connect(lambda: self.useful_info_pop_up("clustering"))

        self.btn_Classification.clicked.connect(lambda: self.next("classification"))
        self.btn_Regression.clicked.connect(lambda: self.next("regression"))
        self.btn_Clustering.clicked.connect(lambda: self.next("clustering"))

    def next(self, event) -> None:
        global_var.prediction_type = event
        next_form = PredictionTypePossibilities.case(event)
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("data_set", "prediction_type")
        last_form = MLTypeWindow(ui_window["model"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class ClassificationSelection(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_KNN.clicked.connect(lambda: self.useful_info_pop_up("knn"))
        self.btn_info_LinearSVC.clicked.connect(lambda: self.useful_info_pop_up("linear_svc"))
        self.btn_info_SVC_rbf.clicked.connect(lambda: self.useful_info_pop_up("svc_rbf"))
        self.btn_info_Gaussian_Naive_Bayes.clicked.connect(lambda: self.useful_info_pop_up("gaussian_naive_bayes"))

        self.btn_LinearSVC.clicked.connect(lambda: self.next("LinearSVC"))
        self.btn_SVC_rbf.clicked.connect(lambda: self.next("SVC"))
        self.btn_KNN.clicked.connect(lambda: self.next("KNeighborsClassifier"))
        self.btn_Gaussian_Naive_Bayes.clicked.connect(lambda: self.next("GaussianNB"))

    def next(self, event) -> None:
        estimator_creator = EstimatorCreator.get_instance()
        clf = estimator_creator.create_estimator(event)
        global_var.estimator = clf
        next_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("prediction_type", estimator=None)
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class RegressionSelection(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Lasso.clicked.connect(lambda: self.useful_info_pop_up("lasso"))
        self.btn_info_SVR_Linear.clicked.connect(lambda: self.useful_info_pop_up("linear_svr"))
        self.btn_info_SVR_rbf.clicked.connect(lambda: self.useful_info_pop_up("svr_rbf"))
        self.btn_info_SGD.clicked.connect(lambda: self.useful_info_pop_up("sgd"))

        self.btn_Lasso.clicked.connect(lambda: self.next("Lasso"))
        self.btn_SVR_Linear.clicked.connect(lambda: self.next("LinearSVR"))
        self.btn_SVR_rbf.clicked.connect(lambda: self.next("SVR"))
        self.btn_SGD.clicked.connect(lambda: self.next("SGDClassifier"))

    def next(self, event) -> None:
        estimator_creator = EstimatorCreator.get_instance()
        clf = estimator_creator.create_estimator(event)
        global_var.estimator = clf
        next_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("prediction_type", estimator=None)
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class ClusteringSelection(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Affinity_Propagation.clicked.connect(lambda: self.useful_info_pop_up("affinity_propagation"))
        self.btn_info_Minibatch_Kmeans.clicked.connect(lambda: self.useful_info_pop_up("minibatch_kmeans"))
        self.btn_info_Meanshift.clicked.connect(lambda: self.useful_info_pop_up("meanshift"))
        self.btn_info_Kmeans.clicked.connect(lambda: self.useful_info_pop_up("kmeans"))

        self.btn_Affinity_Propagation.clicked.connect(lambda: self.next("AffinityPropagation"))
        self.btn_Minibatch_KMeans.clicked.connect(lambda: self.next("MiniBatchKMeans"))
        self.btn_Meanshift.clicked.connect(lambda: self.next("MeanShift"))
        self.btn_KMeans.clicked.connect(lambda: self.next("KMeans"))

    def next(self, event) -> None:
        estimator_creator = EstimatorCreator.get_instance()
        clf = estimator_creator.create_estimator(event)
        global_var.estimator = clf
        next_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("prediction_type", estimator=None)
        last_form = PredictionType(ui_window["prediction_type"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class WantFeatureSelection(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
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
            next_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("uses_feature_selection", "estimator")
        prediction_type = global_var.prediction_type
        last_form = PredictionTypePossibilities.case(prediction_type)
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class FeatureSelectionMethod(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_FS.clicked.connect(lambda: self.useful_info_pop_up("forward_feature_selection"))
        self.btn_info_BFS.clicked.connect(lambda: self.useful_info_pop_up("backwards_feature_selection"))

        self.btn_FS.clicked.connect(lambda: self.next("FFS"))
        self.btn_BFS.clicked.connect(lambda: self.next("BFS"))

    def next(self, event) -> None:
        feature_selection_creator = FeatureSelectorCreator.get_instance()
        feature_selection_method = feature_selection_creator.create_feature_selector(event)
        global_var.feature_selection_method = feature_selection_method
        next_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("uses_feature_selection", "feature_selection_method")
        last_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class WantHyperparameterSearch(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Search_Hiperparameters.clicked.connect(lambda: self.useful_info_pop_up("parameter_search"))
        self.btn_info_Hiperparameters_By_Hand.clicked.connect(lambda:
                                                              self.useful_info_pop_up("manually_set_parameters"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        if self.rbtn_Search_Hiperparameters.isChecked():
            global_var.uses_parameter_search = True
            next_form = HyperparameterMethod(ui_window["hyperparameter_search_method"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        else:
            global_var.uses_parameter_search = False
            user_selection = global_var.estimator.__class__.__name__
            next_form = EstimatorParametersPossibilities.case(user_selection)
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("uses_feature_selection", "uses_parameter_search")
        last_form = WantFeatureSelection(ui_window["feature_selection"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class HyperparameterMethod(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_Bayesian_Search.clicked.connect(lambda: self.handle_input("Bayesian"))
        self.btn_Gird_Search.clicked.connect(lambda: self.handle_input("Greed"))

        self.btn_info_Bayesian_Search.clicked.connect(lambda: self.useful_info_pop_up("bayesian_search"))
        self.btn_info_Grid_Search.clicked.connect(lambda: self.useful_info_pop_up("grid_search"))

    def next(self) -> None:
        next_form = StepByStepLoad(ui_window["result_screen"])
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def handle_input(self, event) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameter_selection_creator = ParameterSearchCreator.get_instance()
            user_selection = global_var.estimator.__class__.__name__
            if event is "Bayesian":
                parameters = BayesianSearchParametersPossibilities.case(user_selection)
                global_var.parameters = parameters
                parameter_search_method = parameter_selection_creator.create_parameter_selector("BS")
                global_var.parameter_search_method = parameter_search_method
            else:
                parameters = GridSearchParametersPossibilities.case(user_selection)
                global_var.parameters = parameters
                parameter_search_method = parameter_selection_creator.create_parameter_selector("GS")
                global_var.parameter_search_method = parameter_search_method
            self.next()

    def back(self) -> None:
        global_var.reset("uses_parameter_search", "parameter_search_method")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class FinalResult(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.lbl_end.mouseReleaseEvent = self.next

    def close_window(self) -> None:
        super(FinalResult, self).close_window()
        widget.close()

    def next(self, event) -> None:
        event.accept()
        global_var.reset()
        self.close_window()


class AffinityPropagationParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_convergencia.clicked.connect(lambda: self.useful_info_pop_up("APROPAGATION_convergencia"))
        self.btn_info_amortiguacion.clicked.connect(lambda: self.useful_info_pop_up("APROPAGATION_amortiguacion"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("APROPAGATION_semilla_random"))
        self.btn_info_afinidad.clicked.connect(lambda: self.useful_info_pop_up("APROPAGATION_afinidad"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"convergence": int(self.sb_convergencia.value()),
                          "damping": float(self.sb_amortiguacion.value()),
                          "random_state": int(self.sb_semilla_random.value()),
                          "affinity": str(self.cb_afinidad.currentText())}
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class GaussianNBParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_variable_refinamiento.clicked.connect(lambda: self.useful_info_pop_up("GNB_refinamiento"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"var_smoothing": float(self.sb_variable_refinamiento.value())}
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class KMeansParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_clusters.clicked.connect(lambda: self.useful_info_pop_up("KMEANS_n_clusters"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("KMEANS_toleracia"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("KMEANS_semilla_random"))
        self.btn_info_algoritmo.clicked.connect(lambda: self.useful_info_pop_up("KMEANS_algoritmo"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"n_clusters": int(self.sb_clusters.value()),
                          "random_state": int(self.sb_semilla_random.value()),
                          "tol": float(self.sb_tolerancia.value()),
                          "algorithm": str(self.cb_algoritmo.currentText())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class KNeighborsClassifierParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_numero_vecinos.clicked.connect(lambda: self.useful_info_pop_up("KNN_n_vecinos"))
        self.btn_info_minkowski_p.clicked.connect(lambda: self.useful_info_pop_up("KNN_p"))
        self.btn_info_tamano_hoja.clicked.connect(lambda: self.useful_info_pop_up("KNN_tamano_hoja"))
        self.btn_info_pesos.clicked.connect(lambda: self.useful_info_pop_up("KNN_pesos"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"n_neighbors": int(self.sb_numero_vecinos.value()),
                          "p": int(self.sb_minkowski_p.value()),
                          "leaf_size": int(self.sb_tamano_hoja.value()),
                          "weights": str(self.cb_pesos.currentText())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class LassoParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_alfa.clicked.connect(lambda: self.useful_info_pop_up("LASSO_alfa"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("LASSO_toleracia"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("LASSO_semilla_random"))
        self.btn_info_seleccion.clicked.connect(lambda: self.useful_info_pop_up("LASSO_seleccion"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"alpha": float(self.sb_alfa.value()),
                          "tol": float(self.sb_tolerancia.value()),
                          "random_state": int(self.sb_semilla_random.value()),
                          "selection": str(self.cb_seleccion.currentText())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class LinearSVCParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("LSVC_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("LSVC_toleracia"))
        self.btn_info_intercepto.clicked.connect(lambda: self.useful_info_pop_up("LSVC_intercepto"))
        self.btn_info_penalidad.clicked.connect(lambda: self.useful_info_pop_up("LSVC_penalidad"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"C": float(self.sb_parametro_regularizacion.value()),
                          "tol": float(self.sb_tolerancia.value()),
                          "intercept_scaling": float(self.sb_intercepto.value()),
                          "penalty": str(self.cb_penalidad.currentText())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class LinearSVRParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("LSVR_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("LSVR_toleracia"))
        self.btn_info_perdida.clicked.connect(lambda: self.useful_info_pop_up("LSVR_perdida"))
        self.btn_info_epsilon.clicked.connect(lambda: self.useful_info_pop_up("LSVR_epsilon"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"C": float(self.sb_parametro_regularizacion.value()),
                          "tol": float(self.sb_tolerancia.value()),
                          "loss": str(self.cb_perdida.currentText()),
                          "epsilon": float(self.sb_epsilon.value())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class MeanShiftParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_ancho_banda.clicked.connect(lambda: self.useful_info_pop_up("MEANSHIFT_ancho_banda"))
        self.btn_info_contenedor_semillas.clicked.connect(lambda:
                                                          self.useful_info_pop_up("MEANSHIFT_contenedor_semilla"))
        self.btn_info_frecuencia_contenedor.clicked.connect(lambda:
                                                            self.useful_info_pop_up("MEANSHIFT_frecuencia_contenedor"))
        self.btn_info_agrupar_todos.clicked.connect(lambda: self.useful_info_pop_up("MEANSHIFT_agrupar_todos"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"bin_seeding": bool(self.cb_contenedor_semillas.currentText()),
                          "cluster_all": bool(self.cb_agrupar_todos.currentText()),
                          "bandwidth": float(self.sb_ancho_banda.value()),
                          "min_bin_freq": int(self.sb_frecuencia_contenedor.value())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class MiniBatchKMeansParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_clusters.clicked.connect(lambda: self.useful_info_pop_up("MINIKMEANS_n_clusters"))
        self.btn_info_tamano_grupo.clicked.connect(lambda: self.useful_info_pop_up("MMINIKMEANS_tamano_grupo"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("MINIKMEANS_semilla_random"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("MINIKMEANS_tolerancia"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"n_clusters": int(self.sb_clusters.value()),
                          "batch_size": int(self.sb_tamano_grupo.value()),
                          "random_state": int(self.sb_semilla_random.value()),
                          "tol": float(self.sb_tolerancia.value())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class SGDClassifierParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_alfa.clicked.connect(lambda: self.useful_info_pop_up("SGD_alfa"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("SGD_tolerancia"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("SGD_semilla_random"))
        self.btn_info_penalidad.clicked.connect(lambda: self.useful_info_pop_up("SGD_penalidad"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"alpha": float(self.sb_alfa.value()),
                          "tol": float(self.sb_tolerancia.value()),
                          "random_state": int(self.sb_semilla_random.value()),
                          "penalty": str(self.cb_penalidad.currentText())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class SVCParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("SVC_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("SVC_tolerancia"))
        self.btn_info_kernel.clicked.connect(lambda: self.useful_info_pop_up("SVC_kernel"))
        self.btn_info_gamma.clicked.connect(lambda: self.useful_info_pop_up("SVC_gamma"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"C": float(self.sb_parametro_regularizacion.value()),
                          "tol": float(self.sb_tolerancia.value()),
                          "kernel": str(self.cb_kernel.currentText()),
                          "gamma": str(self.cb_gamma.currentText())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class SVRParameters(Window):

    def __init__(self, window: str) -> None:
        super().__init__(window)
        self.btn_back.clicked.connect(self.back)

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("SVR_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("SVR_tolerancia"))
        self.btn_info_epsilon.clicked.connect(lambda: self.useful_info_pop_up("SVR_epsilon"))
        self.btn_info_gamma.clicked.connect(lambda: self.useful_info_pop_up("SVR_gamma"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        result = self.last_warning_pop_up()
        if result:
            parameters = {"C": float(self.sb_parametro_regularizacion.value()),
                          "tol": float(self.sb_tolerancia.value()),
                          "epsilon": float(self.sb_epsilon.value()),
                          "gamma": str(self.cb_gamma.currentText())
                          }
            global_var.parameters = parameters
            next_form = StepByStepLoad(ui_window["result_screen"])
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        global_var.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearch(ui_window["hyperparameter_search"])
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


if __name__ == "__main__":
    # global_var instance to store program important variables across all forms
    global_var = GlobalVariables.get_instance()
    # initialize qt resources
    QT_resources.qInitResources()
    # create an app and widget variable to control app logic
    app = QApplication(sys.argv)
    # set app name for all views
    app.setApplicationName("Voorspelling")
    # then change app's icon. There's different sizes if needed
    app_icon = QIcon()
    for key, _ in ui_icons.items():
        app_icon.addFile(ui_icons[key][0], QSize(ui_icons[key][-1], ui_icons[key][-1]))
    app.setWindowIcon(app_icon)
    # QStackedWidget object to control app's views
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
