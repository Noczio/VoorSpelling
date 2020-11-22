from abc import ABC, abstractmethod
from typing import Any
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
import pandas as pd
import numpy as np

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
        clf = BayesSearchCV(estimator=model, search_spaces=parameters, cv=n_folds_validation)
        clf.fit(x, y)
        best_params = clf.best_params_
        return best_params


class GridSearch(ParameterSearch):

    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict,
                          n_folds_validation: int, model: Any) -> dict:
        clf = GridSearchCV(estimator=model, param_grid=parameters, cv=n_folds_validation)
        clf.fit(x, y)
        best_params = clf.best_params_
        return best_params


class ParameterSearchCreator:
    __instance = None
    _types: dict = {"BS": BayesianSearch(), "GS": GridSearch()}

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
        # transform param to capital letters and then replace white spaces
        key = selection_type.upper().replace(" ", "")
        if key in self._types.keys():
            param_search_type = self._types[key]
            return param_search_type
        raise KeyError("Parameter selection key value is wrong. It should be: BS or GS")

    def get_available_types(self) -> tuple:
        available_types = [k for k in self._types.keys()]
        types = tuple(available_types)
        return types
