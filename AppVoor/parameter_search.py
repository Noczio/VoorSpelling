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
    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict, n_folds_validation: int, model: Any):
        pass


class BayesianSearch(ParameterSearch):

    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict, n_folds_validation: int, model: Any):
        clf = BayesSearchCV(model, parameters)
        clf.fit(x, y)


class GridSearch(ParameterSearch):

    def search_parameters(self, x: DataFrame, y: NpArray, parameters: dict, n_folds_validation: int, model: Any):
        clf = GridSearchCV(model, parameters)
        clf.fit(x, y)
