from resources.backend_scripts.switcher import Switch
from resources.frontend_scripts.view import Window
from resources.ui_path import ui_window
from run import ClassificationSelection, RegressionSelection, ClusteringSelection
from run import LinearSVCParameters, SVCParameters, KNeighborsClassifierParameters, GaussianNBParameters, \
    LinearSVRParameters, SVRParameters, LassoParameters, SGDClassifierParameters, AffinityPropagationParameters, \
    KMeansParameters, MiniBatchKMeansParameters, MeanShiftParameters


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
