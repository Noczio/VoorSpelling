from typing import Any

from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso, SGDClassifier
from sklearn.cluster import SpectralClustering, KMeans, MiniBatchKMeans, MeanShift


class EstimatorCreator:
    __instance = None
    _types: dict = {"LSVC": LinearSVC(), "SVC": SVC(),
                    "KNN": KNeighborsClassifier(), "GNB": GaussianNB(),
                    "LSVR": LinearSVR(), "SVR": SVR(),
                    "LASSO": Lasso(), "SGD": SGDClassifier(),
                    "SPC": SpectralClustering(), "KMEANS": KMeans(),
                    "MINIKMEANS": MiniBatchKMeans(), "MEANSHIFT": MeanShift()}

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
        # replace white spaces
        key = estimator_type.replace(" ", "")
        if key in self._types.keys():
            sklearn_estimator_type = self._types[key]
            return sklearn_estimator_type
        raise ValueError("feature selection type value is wrong. It should be: LSVC, SVC, KNN, GNB, LSVR, SVR, LASSO, "
                         "SGD, SPC, KMEANS, MINIKMEANS or MEANSHIFT")

    def get_available_types(self) -> tuple:
        available_types = [k for k in self._types.keys()]
        types = tuple(available_types)
        return types
