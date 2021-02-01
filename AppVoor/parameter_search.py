from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from switcher import Switch

NpArray = np.ndarray
DataFrame = pd.DataFrame


class ParameterSearch(ABC):

    @abstractmethod
    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict,
                          n_folds_validation: int, model: Any) -> dict:
        pass


class BayesianSearch(ParameterSearch):

    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict,
                          n_folds_validation: int, model: Any) -> dict:
        clf = BayesSearchCV(estimator=model, search_spaces=parameters, cv=n_folds_validation, verbose=10)
        clf.fit(x, y)
        best_params = clf.best_params_
        return best_params


class GridSearch(ParameterSearch):

    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict,
                          n_folds_validation: int, model: Any) -> dict:
        clf = GridSearchCV(estimator=model, param_grid=parameters, cv=n_folds_validation, verbose=10)
        clf.fit(x, y)
        best_params = clf.best_params_
        return best_params


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
        return {'C': Real(1, 100, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'dual': Categorical([False]),
                'penalty': Categorical(['l1', 'l2']),
                'intercept_scaling': Real(1, 50, prior='log-uniform')}

    @staticmethod
    def SVC() -> dict:
        return {'C': Real(1, 100, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'gamma': Categorical(['scale', 'auto']),
                'kernel': Categorical(['rbf', 'poly'])}

    @staticmethod
    def KNeighborsClassifier() -> dict:
        return {'n_neighbors': Integer(1, 50),
                'weights': Categorical(['uniform', 'distance']),
                'leaf_size': Integer(30, 100),
                'p': Integer(1, 30),
                'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute'])}

    @staticmethod
    def GaussianNB() -> dict:
        return {'var_smoothing': Real(0.000000001, 100, prior='log-uniform')}

    @staticmethod
    def LinearSVR() -> dict:
        return {'epsilon': Real(0, 100, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'C': Real(1, 100, prior='log-uniform'),
                'loss': Categorical(['epsilon_insensitive', 'squared_epsilon_insensitive']),
                'dual': Categorical([False])}

    @staticmethod
    def SVR() -> dict:
        return {'gamma': Categorical(['scale', 'auto']),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'C': Real(1, 100, prior='log-uniform'),
                'epsilon': Real(0.1, 100, prior='log-uniform'),
                'kernel': Categorical(['rbf', 'poly'])}

    @staticmethod
    def Lasso() -> dict:
        return {'alpha': Real(1, 100, prior='log-uniform'),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'selection': Categorical(['cyclic', 'random']),
                'positive': Categorical([True, False])}

    @staticmethod
    def SGDClassifier() -> dict:
        return {'penalty': Categorical(['l2', 'l1', 'elasticnet']),
                'alpha': Real(0.0001, 100, prior='log-uniform'),
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
        return {'n_clusters': Integer(1, 100),
                'tol': Real(0.0001, 1, prior='log-uniform'),
                'random_state': Integer(0, 1000),
                'algorithm': Categorical(['auto', 'full', 'elkan'])}

    @staticmethod
    def MiniBatchKMeans() -> dict:
        return {'n_clusters': Integer(1, 100),
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
        return {'C': np.arange(1, 30, 1),
                'tol': np.arange(0.0001, 1, 0.001),
                'dual': (False,),
                'penalty': ('l1', 'l2'),
                'intercept_scaling': np.arange(1, 20, 1)}

    @staticmethod
    def SVC() -> dict:
        return {'C': np.arange(1, 30, 1),
                'tol': np.arange(0.0001, 1, 0.001),
                'gamma': ('scale', 'auto'),
                'kernel': ('rbf', 'poly')}

    @staticmethod
    def KNeighborsClassifier() -> dict:
        return {'n_neighbors': np.arange(1, 30, 1),
                'weights': ('uniform', 'distance'),
                'leaf_size': np.arange(30, 80, 10),
                'p': np.arange(1, 20, 1),
                'algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute')}

    @staticmethod
    def GaussianNB() -> dict:
        return {'var_smoothing': [0.000000001, 10]}

    @staticmethod
    def LinearSVR() -> dict:
        return {'epsilon': np.arange(0, 20, 1),
                'tol': np.arange(0.0001, 1, 0.001),
                'C': np.arange(1, 30, 1),
                'loss': ('epsilon_insensitive', 'squared_epsilon_insensitive'),
                'dual': (False,)}

    @staticmethod
    def SVR() -> dict:
        return {'gamma': ('scale', 'auto'),
                'tol': np.arange(0.0001, 1, 0.001),
                'C': np.arange(1, 30, 1),
                'epsilon': np.arange(0.1, 5, 0.5),
                'kernel': ('rbf', 'poly')}

    @staticmethod
    def Lasso() -> dict:
        return {'alpha': np.arange(1, 20, 1),
                'tol': np.arange(0.0001, 1, 0.001),
                'selection': ('cyclic', 'random'),
                'positive': (True, False)}

    @staticmethod
    def SGDClassifier() -> dict:
        return {'penalty': ('l2', 'l1', 'elasticnet'),
                'alpha': np.arange(0.0001, 5, 0.01),
                'tol': np.arange(0.0001, 1, 0.001),
                'random_state': np.arange(0, 2000, 100)}

    @staticmethod
    def AffinityPropagation() -> dict:
        return {'damping': np.arange(0.5, 1, 0.1),
                'convergence_iter': np.arange(15, 50, 5),
                'affinity': ('euclidean', 'precomputed'),
                'random_state': np.arange(0, 2000, 100)}

    @staticmethod
    def KMeans() -> dict:
        return {'n_clusters': np.arange(1, 30, 1),
                'tol': np.arange(0.0001, 1, 0.001),
                'random_state': np.arange(0, 2000, 100),
                'algorithm': ('auto', 'full', 'elkan')}

    @staticmethod
    def MiniBatchKMeans() -> dict:
        return {'n_clusters': np.arange(1, 30, 1),
                'tol': np.arange(0, 1, 0.001),
                'batch_size': np.arange(100, 500, 50),
                'reassignment_ratio': np.arange(0.01, 5, 0.1),
                'random_state': np.arange(0, 2000, 100)}

    @staticmethod
    def MeanShift() -> dict:
        return {'bin_seeding': (True, False),
                'cluster_all': (True, False),
                'min_bin_freq': np.arange(1, 30, 1)}


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
