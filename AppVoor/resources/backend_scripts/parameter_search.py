from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from resources.backend_scripts.switcher import Switch

NpArray = np.ndarray
DataFrame = pd.DataFrame


class ParameterSearch(ABC):

    @abstractmethod
    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict,
                          n_folds_validation: int, model: Any, score_type: str) -> tuple:
        pass


class BayesianSearch(ParameterSearch):

    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict,
                          n_folds_validation: int, model: Any, score_type: str) -> tuple:
        clf = BayesSearchCV(estimator=model, search_spaces=parameters, cv=n_folds_validation,
                            verbose=10, scoring=score_type)
        clf.fit(x, y)
        best_params = clf.best_params_
        best_score = clf.best_score_
        return best_params, best_score


class GridSearch(ParameterSearch):

    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict,
                          n_folds_validation: int, model: Any, score_type: str) -> tuple:
        clf = GridSearchCV(estimator=model, param_grid=parameters, cv=n_folds_validation,
                           verbose=10, scoring=score_type)
        clf.fit(x, y)
        best_params = clf.best_params_
        best_score = clf.best_score_
        return best_params, best_score


class ParameterSearchPossibilities(Switch):

    @staticmethod
    def BS() -> BayesianSearch:
        return BayesianSearch()

    @staticmethod
    def GS() -> GridSearch:
        return GridSearch()

    @staticmethod
    def BayesianSearch() -> BayesianSearch:
        return BayesianSearch()

    @staticmethod
    def GridSearch() -> GridSearch:
        return GridSearch()


class BayesianSearchParametersPossibilities(Switch):

    @staticmethod
    def LinearSVC() -> dict:
        return {'C': Real(1, 30, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'dual': Categorical([False]),
                'penalty': Categorical(['l1', 'l2']),
                'intercept_scaling': Real(1, 50, prior='log-uniform')}

    @staticmethod
    def SVC() -> dict:
        return {'C': Real(1, 30, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'gamma': Categorical(['scale', 'auto']),
                'kernel': Categorical(['rbf', 'sigmoid'])}

    @staticmethod
    def KNeighborsClassifier() -> dict:
        return {'n_neighbors': Integer(1, 40),
                'weights': Categorical(['uniform', 'distance']),
                'leaf_size': Integer(30, 100),
                'p': Integer(1, 30),
                'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'])}

    @staticmethod
    def GaussianNB() -> dict:
        return {'var_smoothing': Real(0.000000001, 100, prior='log-uniform')}

    @staticmethod
    def LinearSVR() -> dict:
        return {'epsilon': Real(0, 30, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'C': Real(1, 30, prior='log-uniform'),
                'loss': Categorical(['epsilon_insensitive', 'squared_epsilon_insensitive']),
                'dual': Categorical([False])}

    @staticmethod
    def SVR() -> dict:
        return {'gamma': Categorical(['scale', 'auto']),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'C': Real(1, 30, prior='log-uniform'),
                'epsilon': Real(0.1, 30, prior='log-uniform'),
                'kernel': Categorical(['rbf', 'sigmoid'])}

    @staticmethod
    def Lasso() -> dict:
        return {'alpha': Real(1, 40, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'selection': Categorical(['cyclic', 'random']),
                'positive': Categorical([True, False])}

    @staticmethod
    def SGDClassifier() -> dict:
        return {'penalty': Categorical(['l2', 'l1', 'elasticnet']),
                'alpha': Real(0.0001, 40, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'random_state': Integer(0, 1000)}

    @staticmethod
    def AffinityPropagation() -> dict:
        return {'damping': Real(0.5, 1, prior='log-uniform'),
                'convergence_iter': Integer(15, 100),
                'affinity': Categorical(['euclidean', 'precomputed']),
                'random_state': Integer(0, 1000)}

    @staticmethod
    def KMeans() -> dict:
        return {'n_clusters': Integer(1, 50),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'random_state': Integer(0, 1000),
                'algorithm': Categorical(['auto', 'full', 'elkan'])}

    @staticmethod
    def MiniBatchKMeans() -> dict:
        return {'n_clusters': Integer(1, 50),
                'tol': Real(0, 1, prior='log-uniform'),
                'batch_size': Integer(100, 512),
                'reassignment_ratio': Real(0.01, 5, prior='log-uniform'),
                'random_state': Integer(0, 1000)}

    @staticmethod
    def MeanShift() -> dict:
        return {'bin_seeding': Categorical([True, False]),
                'cluster_all': Categorical([True, False]),
                'min_bin_freq': Integer(1, 30)}


class GridSearchParametersPossibilities(Switch):

    @staticmethod
    def LinearSVC() -> dict:
        return {'C': np.arange(1, 32, 5),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] + list(np.arange(1, 5.5, 0.5)),
                'dual': (False,),
                'penalty': ('l1', 'l2'),
                'intercept_scaling': np.arange(1, 22, 5)}

    @staticmethod
    def SVC() -> dict:
        return {'C': np.arange(1, 32, 5),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] + list(np.arange(1, 5.5, 0.5)),
                'gamma': ('scale', 'auto'),
                'kernel': ('rbf', 'sigmoid')}

    @staticmethod
    def KNeighborsClassifier() -> dict:
        return {'n_neighbors': np.arange(1, 32, 5),
                'weights': ('uniform', 'distance'),
                'leaf_size': (30, 50, 70, 100),
                'p': (1, 2, 3, 5, 10, 15),
                'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}

    @staticmethod
    def GaussianNB() -> dict:
        return {'var_smoothing': [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] +
                list(np.arange(1, 101, 1))}

    @staticmethod
    def LinearSVR() -> dict:
        return {'epsilon': np.arange(0, 22, 3),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] + list(np.arange(1, 5.5, 0.5)),
                'C': np.arange(1, 32, 5),
                'loss': ('epsilon_insensitive', 'squared_epsilon_insensitive'),
                'dual': (False,)}

    @staticmethod
    def SVR() -> dict:
        return {'gamma': ('scale', 'auto'),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] + list(np.arange(1, 5.5, 0.5)),
                'C': np.arange(1, 32, 5),
                'epsilon': (0.1, 1, 2, 3, 4, 5),
                'kernel': ('rbf', 'sigmoid')}

    @staticmethod
    def Lasso() -> dict:
        return {'alpha': np.arange(1, 32, 5),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] + list(np.arange(1, 5.5, 0.5)),
                'selection': ('cyclic', 'random'),
                'positive': (True, False)}

    @staticmethod
    def SGDClassifier() -> dict:
        return {'penalty': ('l2', 'l1', 'elasticnet'),
                'alpha': (0.0001, 0.01, 1, 2, 3, 4, 5),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] + list(np.arange(1, 5.5, 0.5)),
                'random_state': np.arange(0, 2500, 500)}

    @staticmethod
    def AffinityPropagation() -> dict:
        return {'damping': np.arange(0.5, 1.1, 0.1),
                'convergence_iter': (15, 30, 40, 50),
                'affinity': ('euclidean', 'precomputed'),
                'random_state': np.arange(0, 2500, 500)}

    @staticmethod
    def KMeans() -> dict:
        return {'n_clusters': np.arange(1, 32, 5),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0] + list(np.arange(1, 5.5, 0.5)),
                'random_state': np.arange(0, 2500, 500),
                'algorithm': ('auto', 'full', 'elkan')}

    @staticmethod
    def MiniBatchKMeans() -> dict:
        return {'n_clusters': np.arange(1, 32, 5),
                'tol': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'batch_size': np.arange(100, 600, 100),
                'reassignment_ratio': (0.01, 0.1, 1, 3, 5),
                'random_state': np.arange(0, 2500, 500)}

    @staticmethod
    def MeanShift() -> dict:
        return {'bin_seeding': (True, False),
                'cluster_all': (True, False),
                'min_bin_freq': np.arange(1, 32, 1)}


class ParameterSearchCreator:
    __instance = None

    @staticmethod
    def get_instance() -> "ParameterSearchCreator":
        """Static access method."""
        if ParameterSearchCreator.__instance is None:
            ParameterSearchCreator()
        return ParameterSearchCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if ParameterSearchCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ParameterSearchCreator.__instance = self

    def create_parameter_selector(self, selection_type: str) -> ParameterSearch:
        try:
            parameter_search_name = selection_type.replace(" ", "")
            parameter_search_method = ParameterSearchPossibilities.case(parameter_search_name)
            return parameter_search_method
        except():
            available_types = self.get_available_types()
            types_as_string = ", ".join(available_types)
            raise AttributeError(f"Parameter value is wrong. "
                                 f"It should be any of the following: {types_as_string}")

    def get_available_types(self) -> tuple:
        available_types = [func for func in dir(ParameterSearchPossibilities)
                           if callable(getattr(ParameterSearchPossibilities, func)) and not
                           (func.startswith("__") or func is "case")]
        return tuple(available_types)
