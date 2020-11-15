from abc import abstractmethod, ABC
from typing import Any

import pandas as pd
import numpy as np

from feature_selection import FeatureSelection
from parameter_search import ParameterSearch
from score import CVScore, CVModelScore
from split_data import SplitterReturner

DataFrame = pd.DataFrame
NpArray = np.ndarray


class MachineLearningModel(ABC):
    _feature_selector: FeatureSelection
    _parameter_selector: ParameterSearch
    _best_features: tuple
    _best_params: dict
    _clf: Any

    def __init__(self):
        self._cv_score: CVModelScore = CVScore()

    @property
    def feature_selector(self) -> FeatureSelection:
        return self._feature_selector

    @feature_selector.setter
    def feature_selector(self, value: FeatureSelection) -> None:
        self._feature_selector = value

    @property
    def parameter_selector(self) -> ParameterSearch:
        return self._parameter_selector

    @parameter_selector.setter
    def parameter_selector(self, value: ParameterSearch) -> None:
        self._parameter_selector = value

    @property
    def best_features(self) -> tuple:
        return self._best_features

    @best_features.setter
    def best_features(self, value: tuple) -> None:
        self._best_features = value

    @property
    def best_params(self) -> dict:
        return self.best_params

    @best_params.setter
    def best_params(self, value: dict) -> None:
        self._best_params = value

    @property
    def estimator(self) -> Any:
        return self._clf

    @estimator.setter
    def estimator(self, value: Any) -> None:
        self._clf = value

    @abstractmethod
    def get_score(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        pass


class SimpleModel(MachineLearningModel):

    def get_score(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        # set clf params
        self.estimator.set_params(self.best_params)
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, "roc_auc", 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class OnlyFeatureSelectionModel(MachineLearningModel):

    def get_score(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        # set clf params
        self.estimator.set_params(self.best_params)
        # get best features
        self.best_features = self.feature_selector.select_features(x, y, self.estimator)
        # x now has only the best features
        x = x[self.best_features]
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, "roc_auc", 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class OnlyParameterSearchModel(MachineLearningModel):

    def get_score(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        # transform best params grid into a simple dict
        self.best_params = self.parameter_selector.search_parameters(x, y, self.best_params, 10, self.estimator)
        # set clf params from the previous search
        self.estimator.set_params(self.best_params)
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, "roc_auc", 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class FeatureAndParameterSearch(MachineLearningModel):

    def get_score(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        self.best_params = self.parameter_selector.search_parameters(x, y, self.best_params, 10, self.estimator)
        self.estimator.set_params(self.best_params)
        self.best_features = self.feature_selector.select_features(x, y, self.estimator)
        x = x[self.best_features]

        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, "roc_auc", 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class ModelTypeCreator:
    __instance = None
    _types: dict = {"SM": SimpleModel(), "FSM": OnlyFeatureSelectionModel(),
                    "PSM": OnlyParameterSearchModel(), "AM": FeatureAndParameterSearch()}

    @staticmethod
    def get_instance() -> "ModelTypeCreator":
        """Static access method."""
        if ModelTypeCreator.__instance is None:
            ModelTypeCreator()
        return ModelTypeCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if ModelTypeCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelTypeCreator.__instance = self

    def create_model(self, feature_selection: bool, parameter_search: bool) -> MachineLearningModel:
        if not feature_selection and not parameter_search:
            return self._types["SM"]
        elif feature_selection and not parameter_search:
            return self._types["FSM"]
        elif not feature_selection and parameter_search:
            return self._types["PSM"]
        else:
            return self._types["AM"]

    def get_available_types(self) -> tuple:
        available_types = [k for k in self._types.keys()]
        types = tuple(available_types)
        return types
