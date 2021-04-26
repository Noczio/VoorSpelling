from typing import Any

from sklearn.cluster import AffinityPropagation, KMeans, MiniBatchKMeans, MeanShift
from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR

from resources.backend_scripts.switcher import Switch


class EstimatorPossibilities(Switch):

    @staticmethod
    def LinearSVC() -> LinearSVC:
        return LinearSVC(verbose=10)

    @staticmethod
    def SVC() -> SVC:
        return SVC(verbose=10)

    @staticmethod
    def KNeighborsClassifier() -> KNeighborsClassifier:
        return KNeighborsClassifier()

    @staticmethod
    def GaussianNB() -> GaussianNB:
        return GaussianNB()

    @staticmethod
    def LinearSVR() -> LinearSVR:
        return LinearSVR(verbose=10)

    @staticmethod
    def SVR() -> SVR:
        return SVR(verbose=10)

    @staticmethod
    def Lasso() -> Lasso:
        return Lasso()

    @staticmethod
    def SGDClassifier() -> SGDClassifier:
        return SGDClassifier(verbose=10)

    @staticmethod
    def AffinityPropagation() -> AffinityPropagation:
        return AffinityPropagation(verbose=10)

    @staticmethod
    def KMeans() -> KMeans:
        return KMeans(verbose=10)

    @staticmethod
    def MiniBatchKMeans() -> MiniBatchKMeans:
        return MiniBatchKMeans(verbose=10)

    @staticmethod
    def MeanShift() -> MeanShift:
        return MeanShift()


class EstimatorCreator:
    __instance = None

    @staticmethod
    def get_instance() -> "EstimatorCreator":
        """Static access method."""
        if EstimatorCreator.__instance is None:
            EstimatorCreator()
        return EstimatorCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if EstimatorCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            EstimatorCreator.__instance = self

    def create_estimator(self, estimator_type: str) -> Any:
        try:
            # replace white spaces
            estimator_name = estimator_type.replace(" ", "")
            # get estimator using a switch
            estimator = EstimatorPossibilities.case(estimator_name)
            return estimator
        except():
            available_types = self.get_available_types()
            types_as_string = ", ".join(available_types)
            raise AttributeError(f"Parameter value is wrong. "
                                 f"It should be any of the following: {types_as_string}")

    def get_available_types(self) -> tuple:
        available_types = [func for func in dir(EstimatorPossibilities)
                           if callable(getattr(EstimatorPossibilities, func)) and not
                           (func.startswith("__") or func is "case")]
        return tuple(available_types)
