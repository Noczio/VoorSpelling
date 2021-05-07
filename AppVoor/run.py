import sys
from typing import Any

import numpy as np
import pandas as pd
from PyQt5.QtCore import QRect, QThreadPool, QThread
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import QApplication

from abc import abstractmethod
from resources.backend_scripts.auto_ml import JarAutoML, AutoExecutioner
from resources.backend_scripts.estimator_creation import EstimatorCreator
from resources.backend_scripts.feature_selection import FeatureSelectorCreator, FeatureSelection
from resources.backend_scripts.load_data import LoaderCreator
from resources.backend_scripts.model_creation import SBSModelCreator
from resources.backend_scripts.parameter_search import BayesianSearchParametersPossibilities
from resources.backend_scripts.parameter_search import GridSearchParametersPossibilities
from resources.backend_scripts.parameter_search import ParameterSearchCreator, ParameterSearch
from resources.backend_scripts.result_creation import FCreator, SBSResult
from resources.backend_scripts.switcher import Switch
from resources.frontend_scripts.modified_widgets import QDragAndDropButton, QLoadButton
from resources.frontend_scripts.parallel import LongWorker, EmittingStream
from resources.frontend_scripts.pop_up import PopUp, WarningPopUp, CriticalPopUp
from resources.frontend_scripts.view import Window
from resources.integration.main import MainInitializer
from resources.integration.other.cancel_stylesheet import cancel_buttons_style
from resources.integration.other.load_errors import load_dataset_errors
from resources.integration.other.load_stylesheet import load_buttons_style
from resources.integration.other.ui_path import ui_window, ui_welcome_message
from resources.json_info.welcome import WelcomeMessenger

DataFrame = pd.DataFrame
NpArray = np.ndarray


class PredictionTypePossibilities(Switch):
    """Switch implementation to open a Classification, Regression or Clustering Window based on user's input"""

    @staticmethod
    def Classification() -> Window:
        return ClassificationSelectionWindow()

    @staticmethod
    def Regression() -> Window:
        return RegressionSelectionWindow()

    @staticmethod
    def Clustering() -> Window:
        return ClusteringSelectionWindow()


class EstimatorParametersPossibilities(Switch):
    """Switch implementation to open a ByHandParametersWindow based on user's input"""

    @staticmethod
    def LinearSVC() -> Window:
        return LinearSVCParametersWindow()

    @staticmethod
    def SVC() -> Window:
        return SVCParametersWindow()

    @staticmethod
    def KNeighborsClassifier() -> Window:
        return KNeighborsClassifierParametersWindow()

    @staticmethod
    def GaussianNB() -> Window:
        return GaussianNBParametersWindow()

    @staticmethod
    def LinearSVR() -> Window:
        return LinearSVRParametersWindow()

    @staticmethod
    def SVR() -> Window:
        return SVRParametersWindow()

    @staticmethod
    def Lasso() -> Window:
        return LassoParametersWindow()

    @staticmethod
    def SGDClassifier() -> Window:
        return SGDClassifierParametersWindow()

    @staticmethod
    def AffinityPropagation() -> Window:
        return AffinityPropagationParametersWindow()

    @staticmethod
    def KMeans() -> Window:
        return KMeansParametersWindow()

    @staticmethod
    def MiniBatchKMeans() -> Window:
        return MiniBatchKMeansParametersWindow()

    @staticmethod
    def MeanShift() -> Window:
        return MeanShiftParametersWindow()


class HomeWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Home"])
        self.btn_next.clicked.connect(self.next)

    def on_load(self) -> None:
        super(HomeWindow, self).on_load()
        messenger = WelcomeMessenger(ui_welcome_message["Path"])
        text = str(messenger)
        self.lbl_description.setText(text)

    def centered(self) -> None:
        frame_gm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(QApplication.desktop().cursor().pos())
        center_point = QApplication.desktop().screenGeometry(screen).center()
        frame_gm.moveCenter(center_point)
        widget.move(frame_gm.topLeft())

    def next(self) -> None:
        next_form = DataSetWindow()
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class DataSetWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Dataset"])
        # by default _df_file_type is CSV and _last_btn_used is any. Also, btn_tsv must be hidden
        self._df_file_type: str = "CSV"
        self._last_btn_used: str = "Any"
        self.btn_tsv.hide()
        # initialize btn_drag_file and btn_load_file variables and then set their styles
        self.btn_drag_file = QDragAndDropButton(self.main_area)
        self.btn_load_file = QLoadButton(self.main_area)
        self._set_load_and_drag_buttons()
        # change _df_file_type to the other when clicked
        self.btn_csv.clicked.connect(lambda: self._set_df_file_type("TSV"))
        self.btn_tsv.clicked.connect(lambda: self._set_df_file_type("CSV"))

        self.btn_load_file.loaded.connect(lambda: self._set_last_emitted("Load"))
        self.btn_drag_file.loaded.connect(lambda: self._set_last_emitted("Drag"))
        self.btn_next.clicked.connect(self._handle_file)

        self.btn_info_data_type.clicked.connect(lambda: self.useful_info_pop_up("File_separation"))
        self.btn_back.clicked.connect(self.back)

    def _set_load_and_drag_buttons(self) -> None:
        font = QFont()
        font.setPointSize(14)
        btn_load_style = load_buttons_style[self._last_btn_used][0]
        btn_drag_style = load_buttons_style[self._last_btn_used][-1]
        # set btn_drag_style for btn_drag_file
        self.btn_drag_file.setObjectName(u"btn_drag_file")
        self.btn_drag_file.setGeometry(QRect(360, 350, 331, 191))
        self.btn_drag_file.setStyleSheet(btn_drag_style)
        self.btn_drag_file.setText("")
        self.btn_drag_file.raise_()
        # set btn_drag_style for btn_load_file
        self.btn_load_file.setObjectName(u"btn_load_file")
        self.btn_load_file.setGeometry(QRect(410, 490, 230, 40))
        self.btn_load_file.setFont(font)
        self.btn_load_file.setStyleSheet(btn_load_style)
        self.btn_load_file.setText("Buscar archivo")
        self.btn_load_file.raise_()

    def _set_df_file_type(self, event: str) -> None:
        self._df_file_type = event

    def _set_last_emitted(self, event: str) -> None:
        self._last_btn_used = event
        btn_load_style = load_buttons_style[self._last_btn_used][0]
        btn_drag_style = load_buttons_style[self._last_btn_used][-1]
        btn_load_file_path = self.btn_load_file.file_path
        btn_drag_file_path = self.btn_drag_file.file_path
        if btn_load_file_path is not "" or btn_drag_file_path is not "":
            self.btn_load_file.setStyleSheet(btn_load_style)
            self.btn_drag_file.setStyleSheet(btn_drag_style)

    def next(self) -> None:
        next_form = MLTypeWindow()
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def last_warning_pop_up(self, body: str, additional: str) -> bool:
        critical_pop_up: PopUp = CriticalPopUp()
        critical_pop_up.open_pop_up("Error", body, additional)
        return True

    def handle_error(self, error: Exception) -> None:
        error_data = load_dataset_errors[type(error)]
        self.last_warning_pop_up(error_data["Body"], error_data["Additional"])

    def _handle_file(self) -> None:
        try:
            if self._last_btn_used is "Any":
                body = "Ningún archivo ha sido seleccionado. Por favor subir un archivo y seleccionar la separación " \
                       "del mismo, ya sea TSV o CSV"
                self.last_warning_pop_up(body, "")
            else:
                btn_load_file_file_path = self.btn_load_file.file_path
                btn_drag_file_file_path = self.btn_drag_file.file_path
                file_path = btn_load_file_file_path if self._last_btn_used is "Load" else btn_drag_file_file_path
                loader = LoaderCreator.create_loader(file_path, self._df_file_type)
                data_frame = loader.get_file_transformed()
                variables.data_frame = data_frame
                self.next()
        except Exception as error:
            self.handle_error(error)

    def back(self) -> None:
        variables.reset("data_set")
        last_form = HomeWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class MLTypeWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Model"])
        self.btn_back.clicked.connect(self.back)

        self.btn_info_automl.clicked.connect(lambda: self.useful_info_pop_up("Auto_machine_learning"))
        self.btn_info_sbsml.clicked.connect(lambda: self.useful_info_pop_up("Step_by_step"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        if self.rbtn_sbsml.isChecked():
            next_form = PredictionTypeWindow()
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        else:
            want_to_start_training = self.last_warning_pop_up()
            if want_to_start_training:
                next_form = AutoTrainingWindow()
                widget.addWidget(next_form)
                widget.removeWidget(widget.currentWidget())
                widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        variables.reset("data_set")
        last_form = DataSetWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class TrainingWindow(Window):
    """Abstraction  for training view based on Window class"""

    def __init__(self, window_path: str) -> None:
        super().__init__(window_path)
        sys.stdout = EmittingStream(textWritten=self._add_info)  # textWritten works this way
        self.lbl_cancel.mouseReleaseEvent = self._cancel_training
        # thread_pool and ml_worker setup. A training view needs 2 threads to work properly
        self._thread_pool = QThreadPool()
        self._thread_pool.setMaxThreadCount(2)
        self._ml_worker = LongWorker()
        self._ml_worker.set_params(self._train_model)
        self._ml_worker.signals.program_finished.connect(self.next)
        self._ml_worker.signals.program_error.connect(self.handle_error)
        self._ml_worker.signals.result.connect(self._add_info)
        self._ml_worker.setAutoDelete(True)
        self._thread_pool.start(self._ml_worker, priority=1)

    def close_window(self) -> None:
        super(TrainingWindow, self).close_window()
        widget.close()

    def next(self) -> None:
        QThread.sleep(1)
        next_form = FinalResultWindow()
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    @abstractmethod
    def _train_model(self) -> None:
        pass

    def last_warning_pop_up(self) -> bool:
        pop_up: PopUp = WarningPopUp()
        title = "Cancelar entrenamiento"
        body = "¿Estas seguro que deseas cancelar el entrenamiento?"
        additional = "La aplicación se cerrará para evitar conflictos con las variables utilizadas hasta el momento."
        answer = pop_up.open_pop_up(title, body, additional)
        return answer

    def _cancel_training(self, event) -> None:
        """Show a Warning pop up and then if user wants to finished the app, close it"""
        event.accept()
        want_to_stop_training = self.last_warning_pop_up()
        if want_to_stop_training:
            self._thread_pool.cancel(self._ml_worker)
            self.close_window()

    def handle_error(self, error) -> None:
        """Print error message to the QTextEdit"""

        def write_error():
            for i in info:
                self._add_info(i)
                QThread.sleep(1)

        # deactivate lbl press behaviour due to an error
        lbl_cancel_style = cancel_buttons_style["Not_available"]
        self.lbl_cancel.mouseReleaseEvent = None
        self.lbl_cancel.setStyleSheet(lbl_cancel_style)
        info = ("Error\n", str(error), "\nCerrando aplicación para evitar conflictos de memoria" + " ", ".", ".", ".")
        # worker to write info to ted_info
        temp_worker = LongWorker(func=write_error)
        temp_worker.signals.program_finished.connect(self.close_window)
        self._thread_pool.start(temp_worker, priority=2)

    def _add_info(self, info: str) -> None:
        """Append text to the QTextEdit."""
        cursor = self.ted_info.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(info)
        self.ted_info.setTextCursor(cursor)
        self.ted_info.ensureCursorVisible()


class AutoTrainingWindow(TrainingWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["Result_screen"])

    def _train_model(self) -> None:
        automl_ml = JarAutoML(10, False, 5000)
        model = AutoExecutioner(automl_ml)
        data_frame = variables.data_frame
        model.train_model(data_frame)


class StepByStepTrainingWindow(TrainingWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["Result_screen"])

    def _save_results(self, score_text: str, estimator: Any, initial_parameters: dict, best_features: list,
                      best_parameters: dict, feature_selector: FeatureSelection, parameter_selector: ParameterSearch,
                      folder_path: str) -> None:
        # All important info is storage in a variable to be displayed in a markdown file as a 2x5 table
        feature_selection_method_name = "No" if feature_selector is None else feature_selector.__class__.__name__
        parameter_search_method_name = "No" if parameter_selector is None else parameter_selector.__class__.__name__
        prediction_type_name = variables.prediction_type
        estimator_name = estimator.__class__.__name__

        info = ["Opción", "SBS Voorspelling",
                "Tipo de predicción", prediction_type_name,
                "Estimador", estimator_name,
                "Selección de características", feature_selection_method_name,
                "Selección de hiperparámetros", parameter_search_method_name
                ]
        table = {"columns": 2, "rows": 5, "info": info}

        # save estimator info results into markdown file
        SBSResult.estimator_info(table, best_features, initial_parameters, best_parameters, score_text, folder_path)
        SBSResult.dump_estimator(estimator, folder_path)
        # Finally, after all is finished write ted info to its markdown file
        ted_text = self.ted_info.toPlainText()
        fixed_ted_text = ted_text.split("\n")
        SBSResult.console_info(fixed_ted_text, folder_path)

    def _train_model(self) -> None:
        uses_feature_selection: bool = variables.uses_feature_selection
        uses_parameter_search: bool = variables.uses_parameter_search

        # get important info and then score model
        model = SBSModelCreator.create_model(uses_feature_selection, uses_parameter_search)
        model.estimator = variables.estimator
        model.initial_parameters = variables.parameters
        model.feature_selector = variables.feature_selection_method
        model.parameter_selector = variables.parameter_search_method
        model.data_frame = variables.data_frame

        # scoring metric by default for each type of prediction
        score_type = {"Classification": "accuracy",
                      "Regression": "neg_mean_squared_error",
                      "Clustering": "mutual_info_score"}
        prediction_type = variables.prediction_type
        # score model, then get a user friendly message for that score and finally return data
        score = model.score_model(score_type[prediction_type], 10)
        score_text = f"{score_type[prediction_type]}: {score}"

        f_creator = FCreator()
        folder_path = f_creator.folder_path
        self._save_results(score_text, model.estimator, model.initial_parameters, list(model.best_features),
                           model.best_parameters, model.feature_selector, model.parameter_selector, folder_path)


class PredictionTypeWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Prediction_type"])
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Classification.clicked.connect(lambda: self.useful_info_pop_up("Classification"))
        self.btn_info_Regression.clicked.connect(lambda: self.useful_info_pop_up("Regression"))
        self.btn_info_Clustering.clicked.connect(lambda: self.useful_info_pop_up("Clustering"))

        self.btn_Classification.clicked.connect(lambda: self.next("Classification"))
        self.btn_Regression.clicked.connect(lambda: self.next("Regression"))
        self.btn_Clustering.clicked.connect(lambda: self.next("Clustering"))

    def next(self, event: str) -> None:
        variables.prediction_type = event
        next_form = PredictionTypePossibilities.case(event)
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        variables.reset("data_set", "prediction_type")
        last_form = MLTypeWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class EstimatorSelectionWindow(Window):
    """Abstraction for estimator's view based on Window class"""

    def __init__(self, window_path: str) -> None:
        super().__init__(window_path)
        self.btn_back.clicked.connect(self.back)

    def next(self, event: str) -> None:
        estimator = EstimatorCreator.create_estimator(event)
        variables.estimator = estimator
        next_form = WantFeatureSelectionWindow()
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        variables.reset("prediction_type", estimator=None)
        last_form = PredictionTypeWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class ClassificationSelectionWindow(EstimatorSelectionWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["Classification"])

        self.btn_info_KNN.clicked.connect(lambda: self.useful_info_pop_up("KNeighborsClassifier"))
        self.btn_info_LinearSVC.clicked.connect(lambda: self.useful_info_pop_up("LinearSVC"))
        self.btn_info_SVC_rbf.clicked.connect(lambda: self.useful_info_pop_up("SVC"))
        self.btn_info_Gaussian_Naive_Bayes.clicked.connect(lambda: self.useful_info_pop_up("GaussianNB"))

        self.btn_KNN.clicked.connect(lambda: self.next("KNeighborsClassifier"))
        self.btn_LinearSVC.clicked.connect(lambda: self.next("LinearSVC"))
        self.btn_SVC_rbf.clicked.connect(lambda: self.next("SVC"))
        self.btn_Gaussian_Naive_Bayes.clicked.connect(lambda: self.next("GaussianNB"))


class RegressionSelectionWindow(EstimatorSelectionWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["Regression"])

        self.btn_info_Lasso.clicked.connect(lambda: self.useful_info_pop_up("Lasso"))
        self.btn_info_SVR_Linear.clicked.connect(lambda: self.useful_info_pop_up("LinearSVR"))
        self.btn_info_SVR_rbf.clicked.connect(lambda: self.useful_info_pop_up("SVR"))
        self.btn_info_SGD.clicked.connect(lambda: self.useful_info_pop_up("SGDClassifier"))

        self.btn_Lasso.clicked.connect(lambda: self.next("Lasso"))
        self.btn_SVR_Linear.clicked.connect(lambda: self.next("LinearSVR"))
        self.btn_SVR_rbf.clicked.connect(lambda: self.next("SVR"))
        self.btn_SGD.clicked.connect(lambda: self.next("SGDClassifier"))


class ClusteringSelectionWindow(EstimatorSelectionWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["Clustering"])

        self.btn_info_Affinity_Propagation.clicked.connect(lambda: self.useful_info_pop_up("AffinityPropagation"))
        self.btn_info_Minibatch_Kmeans.clicked.connect(lambda: self.useful_info_pop_up("MiniBatchKMeans"))
        self.btn_info_Meanshift.clicked.connect(lambda: self.useful_info_pop_up("MeanShift"))
        self.btn_info_Kmeans.clicked.connect(lambda: self.useful_info_pop_up("KMeans"))

        self.btn_Affinity_Propagation.clicked.connect(lambda: self.next("AffinityPropagation"))
        self.btn_Minibatch_KMeans.clicked.connect(lambda: self.next("MiniBatchKMeans"))
        self.btn_Meanshift.clicked.connect(lambda: self.next("MeanShift"))
        self.btn_KMeans.clicked.connect(lambda: self.next("KMeans"))


class WantFeatureSelectionWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Feature_selection"])
        self.btn_back.clicked.connect(self.back)

        self.btn_info_FSM.clicked.connect(lambda: self.useful_info_pop_up("Feature_selection"))
        self.btn_info_No_FSM.clicked.connect(lambda: self.useful_info_pop_up("No_feature_selection"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        if self.rbtn_FSM.isChecked():
            variables.uses_feature_selection = True
            next_form = FeatureSelectionMethodWindow()
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        else:
            variables.uses_feature_selection = False
            next_form = WantHyperparameterSearchWindow()
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        variables.reset("uses_feature_selection", "estimator")
        prediction_type = variables.prediction_type
        last_form = PredictionTypePossibilities.case(prediction_type)
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class FeatureSelectionMethodWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Feature_selection_method"])
        self.btn_back.clicked.connect(self.back)

        self.btn_info_FS.clicked.connect(lambda: self.useful_info_pop_up("Forward_feature_selection"))
        self.btn_info_BFS.clicked.connect(lambda: self.useful_info_pop_up("Backwards_feature_selection"))

        self.btn_FS.clicked.connect(lambda: self.next("FFS"))
        self.btn_BFS.clicked.connect(lambda: self.next("BFS"))

    def next(self, event: str) -> None:
        feature_selection_method = FeatureSelectorCreator.create_feature_selector(event)
        variables.feature_selection_method = feature_selection_method
        next_form = WantHyperparameterSearchWindow()
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        variables.reset("uses_feature_selection", "feature_selection_method")
        last_form = WantFeatureSelectionWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class WantHyperparameterSearchWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Hyperparameter_search"])
        self.btn_back.clicked.connect(self.back)

        self.btn_info_Search_Hiperparameters.clicked.connect(lambda: self.useful_info_pop_up("Parameter_search"))
        self.btn_info_Hiperparameters_By_Hand.clicked.connect(lambda:
                                                              self.useful_info_pop_up("Manually_set_parameters"))

        self.btn_next.clicked.connect(self.next)

    def next(self) -> None:
        if self.rbtn_Search_Hiperparameters.isChecked():
            variables.uses_parameter_search = True
            next_form = HyperparameterMethodWindow()
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())
        else:
            variables.uses_parameter_search = False
            user_selection = variables.estimator.__class__.__name__
            next_form = EstimatorParametersPossibilities.case(user_selection)
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        variables.reset("uses_feature_selection", "uses_parameter_search")
        last_form = WantFeatureSelectionWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class HyperparameterMethodWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Hyperparameter_search_method"])
        self.btn_back.clicked.connect(self.back)

        self.btn_Bayesian_Search.clicked.connect(lambda: self.handle_input("BS"))
        self.btn_Gird_Search.clicked.connect(lambda: self.handle_input("GS"))

        self.btn_info_Bayesian_Search.clicked.connect(lambda: self.useful_info_pop_up("Bayesian_search"))
        self.btn_info_Grid_Search.clicked.connect(lambda: self.useful_info_pop_up("Grid_search"))

    def next(self) -> None:
        next_form = StepByStepTrainingWindow()
        widget.addWidget(next_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())

    def handle_input(self, event: str) -> None:
        parameter_search_selected_by_user: str = variables.estimator.__class__.__name__
        want_to_start_training: bool = self.last_warning_pop_up()
        if want_to_start_training and event is "BS":
            variables.parameters = BayesianSearchParametersPossibilities.case(parameter_search_selected_by_user)
            variables.parameter_search_method = ParameterSearchCreator.create_parameter_selector(event)
            self.next()
        elif want_to_start_training and event is "GS":
            variables.parameters = GridSearchParametersPossibilities.case(parameter_search_selected_by_user)
            variables.parameter_search_method = ParameterSearchCreator.create_parameter_selector(event)
            self.next()

    def back(self) -> None:
        variables.reset("uses_parameter_search", "parameter_search_method")
        last_form = WantHyperparameterSearchWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class FinalResultWindow(Window):

    def __init__(self) -> None:
        super().__init__(ui_window["Result_final"])
        self.lbl_end.mouseReleaseEvent = self.next

    def close_window(self) -> None:
        super(FinalResultWindow, self).close_window()
        widget.close()

    def next(self, event) -> None:
        event.accept()
        variables.reset()
        self.close_window()


class ByHandParametersWindow(Window):

    def __init__(self, window_path: str) -> None:
        super().__init__(window_path)
        self.btn_back.clicked.connect(self.back)

    def next(self, parameters: dict) -> None:
        want_to_start_training = self.last_warning_pop_up()
        if want_to_start_training:
            variables.parameters = parameters
            next_form = StepByStepTrainingWindow()
            widget.addWidget(next_form)
            widget.removeWidget(widget.currentWidget())
            widget.setCurrentIndex(widget.currentIndex())

    def back(self) -> None:
        variables.reset("parameters", "uses_parameter_search")
        last_form = WantHyperparameterSearchWindow()
        widget.addWidget(last_form)
        widget.removeWidget(widget.currentWidget())
        widget.setCurrentIndex(widget.currentIndex())


class AffinityPropagationParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["AffinityPropagation"])

        self.btn_info_convergencia.clicked.connect(lambda: self.useful_info_pop_up("AffinityPropagation_convergencia"))
        self.btn_info_amortiguacion.clicked.connect(
            lambda: self.useful_info_pop_up("AffinityPropagation_amortiguacion"))
        self.btn_info_semilla_random.clicked.connect(
            lambda: self.useful_info_pop_up("AffinityPropagation_semilla_random"))
        self.btn_info_afinidad.clicked.connect(lambda: self.useful_info_pop_up("AffinityPropagation_afinidad"))

        self.btn_next.clicked.connect(lambda: self.next({"convergence": int(self.sb_convergencia.value()),
                                                         "damping": float(self.sb_amortiguacion.value()),
                                                         "random_state": int(self.sb_semilla_random.value()),
                                                         "affinity": str(self.cb_afinidad.currentText())}))


class GaussianNBParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["GaussianNB"])

        self.btn_info_variable_refinamiento.clicked.connect(lambda: self.useful_info_pop_up("GaussianNB_refinamiento"))

        self.btn_next.clicked.connect(
            lambda: self.next({"var_smoothing": float(self.sb_variable_refinamiento.value())}))


class KMeansParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["KMeans"])

        self.btn_info_clusters.clicked.connect(lambda: self.useful_info_pop_up("KMeans_n_clusters"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("KMeans_toleracia"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("KMeans_semilla_random"))
        self.btn_info_algoritmo.clicked.connect(lambda: self.useful_info_pop_up("KMeans_algoritmo"))

        self.btn_next.clicked.connect(lambda: self.next({"n_clusters": int(self.sb_clusters.value()),
                                                         "random_state": int(self.sb_semilla_random.value()),
                                                         "tol": float(self.sb_tolerancia.value()),
                                                         "algorithm": str(self.cb_algoritmo.currentText())
                                                         }))


class KNeighborsClassifierParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["KNeighborsClassifier"])

        self.btn_info_numero_vecinos.clicked.connect(lambda: self.useful_info_pop_up("KNeighborsClassifier_n_vecinos"))
        self.btn_info_minkowski_p.clicked.connect(lambda: self.useful_info_pop_up("KNeighborsClassifier_p"))
        self.btn_info_tamano_hoja.clicked.connect(lambda: self.useful_info_pop_up("KNeighborsClassifier_tamano_hoja"))
        self.btn_info_pesos.clicked.connect(lambda: self.useful_info_pop_up("KNeighborsClassifier_pesos"))

        self.btn_next.clicked.connect(lambda: self.next({"n_neighbors": int(self.sb_numero_vecinos.value()),
                                                         "p": int(self.sb_minkowski_p.value()),
                                                         "leaf_size": int(self.sb_tamano_hoja.value()),
                                                         "weights": str(self.cb_pesos.currentText())
                                                         }))


class LassoParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["Lasso"])

        self.btn_info_alfa.clicked.connect(lambda: self.useful_info_pop_up("Lasso_alfa"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("Lasso_toleracia"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("Lasso_semilla_random"))
        self.btn_info_seleccion.clicked.connect(lambda: self.useful_info_pop_up("Lasso_seleccion"))

        self.btn_next.clicked.connect(lambda: self.next({"alpha": float(self.sb_alfa.value()),
                                                         "tol": float(self.sb_tolerancia.value()),
                                                         "random_state": int(self.sb_semilla_random.value()),
                                                         "selection": str(self.cb_seleccion.currentText())
                                                         }))


class LinearSVCParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["LinearSVC"])

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("LinearSVC_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("LinearSVC_toleracia"))
        self.btn_info_intercepto.clicked.connect(lambda: self.useful_info_pop_up("LinearSVC_intercepto"))
        self.btn_info_penalidad.clicked.connect(lambda: self.useful_info_pop_up("LinearSVC_penalidad"))

        self.btn_next.clicked.connect(lambda: self.next({"C": float(self.sb_parametro_regularizacion.value()),
                                                         "tol": float(self.sb_tolerancia.value()),
                                                         "intercept_scaling": float(self.sb_intercepto.value()),
                                                         "penalty": str(self.cb_penalidad.currentText())
                                                         }))


class LinearSVRParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["LinearSVR"])

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("LinearSVR_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("LinearSVRtoleracia"))
        self.btn_info_perdida.clicked.connect(lambda: self.useful_info_pop_up("LinearSVR_perdida"))
        self.btn_info_epsilon.clicked.connect(lambda: self.useful_info_pop_up("LinearSVR_epsilon"))

        self.btn_next.clicked.connect(lambda: self.next({"C": float(self.sb_parametro_regularizacion.value()),
                                                         "tol": float(self.sb_tolerancia.value()),
                                                         "loss": str(self.cb_perdida.currentText()),
                                                         "epsilon": float(self.sb_epsilon.value())
                                                         }))


class MeanShiftParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["MeanShift"])

        self.btn_info_ancho_banda.clicked.connect(lambda: self.useful_info_pop_up("MeanShift_ancho_banda"))
        self.btn_info_contenedor_semillas.clicked.connect(lambda:
                                                          self.useful_info_pop_up("MeanShift_contenedor_semilla"))
        self.btn_info_frecuencia_contenedor.clicked.connect(lambda:
                                                            self.useful_info_pop_up("MeanShift_frecuencia_contenedor"))
        self.btn_info_agrupar_todos.clicked.connect(lambda: self.useful_info_pop_up("MeanShift_agrupar_todos"))

        self.btn_next.clicked.connect(lambda: self.next({"bin_seeding": bool(self.cb_contenedor_semillas.currentText()),
                                                         "cluster_all": bool(self.cb_agrupar_todos.currentText()),
                                                         "bandwidth": float(self.sb_ancho_banda.value()),
                                                         "min_bin_freq": int(self.sb_frecuencia_contenedor.value())
                                                         }))


class MiniBatchKMeansParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["MiniBatchKMeans"])

        self.btn_info_clusters.clicked.connect(lambda: self.useful_info_pop_up("MiniBatchKMeans_n_clusters"))
        self.btn_info_tamano_grupo.clicked.connect(lambda: self.useful_info_pop_up("MiniBatchKMeans_tamano_grupo"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("MiniBatchKMeans_semilla_random"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("MiniBatchKMeans_tolerancia"))

        self.btn_next.clicked.connect(lambda: self.next({"n_clusters": int(self.sb_clusters.value()),
                                                         "batch_size": int(self.sb_tamano_grupo.value()),
                                                         "random_state": int(self.sb_semilla_random.value()),
                                                         "tol": float(self.sb_tolerancia.value())
                                                         }))


class SGDClassifierParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["SGDClassifier"])

        self.btn_info_alfa.clicked.connect(lambda: self.useful_info_pop_up("SGDClassifier_alfa"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("SGDClassifier_tolerancia"))
        self.btn_info_semilla_random.clicked.connect(lambda: self.useful_info_pop_up("SGDClassifier_semilla_random"))
        self.btn_info_penalidad.clicked.connect(lambda: self.useful_info_pop_up("SGDClassifier_penalidad"))

        self.btn_next.clicked.connect(lambda: self.next({"alpha": float(self.sb_alfa.value()),
                                                         "tol": float(self.sb_tolerancia.value()),
                                                         "random_state": int(self.sb_semilla_random.value()),
                                                         "penalty": str(self.cb_penalidad.currentText())
                                                         }))


class SVCParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["SVC"])

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("SVC_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("SVC_tolerancia"))
        self.btn_info_kernel.clicked.connect(lambda: self.useful_info_pop_up("SVC_kernel"))
        self.btn_info_gamma.clicked.connect(lambda: self.useful_info_pop_up("SVC_gamma"))

        self.btn_next.clicked.connect(lambda: self.next({"C": float(self.sb_parametro_regularizacion.value()),
                                                         "tol": float(self.sb_tolerancia.value()),
                                                         "kernel": str(self.cb_kernel.currentText()),
                                                         "gamma": str(self.cb_gamma.currentText())
                                                         }))


class SVRParametersWindow(ByHandParametersWindow):

    def __init__(self) -> None:
        super().__init__(ui_window["SVR"])

        self.btn_info_parametro_regularizacion.clicked.connect(lambda: self.useful_info_pop_up("SVR_C"))
        self.btn_info_tolerancia.clicked.connect(lambda: self.useful_info_pop_up("SVR_tolerancia"))
        self.btn_info_epsilon.clicked.connect(lambda: self.useful_info_pop_up("SVR_epsilon"))
        self.btn_info_gamma.clicked.connect(lambda: self.useful_info_pop_up("SVR_gamma"))

        self.btn_next.clicked.connect(lambda: self.next({"C": float(self.sb_parametro_regularizacion.value()),
                                                         "tol": float(self.sb_tolerancia.value()),
                                                         "epsilon": float(self.sb_epsilon.value()),
                                                         "gamma": str(self.cb_gamma.currentText())
                                                         }))


if __name__ == "__main__":
    main = MainInitializer()
    app, widget, variables = main.program_resources()
    # first window/view is HomeWindow. Create an instance, then center it on user's screen
    first_window = HomeWindow()
    first_window.centered()
    # add first_window to the QStackedWidget, then show it and start app's loop
    widget.addWidget(first_window)
    widget.show()
    sys.exit(app.exec_())
