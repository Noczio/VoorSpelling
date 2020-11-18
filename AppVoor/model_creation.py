from abc import abstractmethod, ABC
from typing import Any

import pandas as pd
import numpy as np
from sklearn.base import clone

from feature_selection import FeatureSelection
from parameter_search import ParameterSearch
from score import CVScore, CVModelScore
from split_data import SplitterReturner

DataFrame = pd.DataFrame
NpArray = np.ndarray


class MachineLearningModel(ABC):
    _feature_selector: FeatureSelection = None
    _parameter_selector: ParameterSearch = None
    _best_features: NpArray = None
    _best_params: dict = None
    _initial_params: dict = None
    _clf: Any = None
    _cv_score: CVModelScore = CVScore()

    def __init__(self):
        pass

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
    def best_features(self) -> NpArray:
        return self._best_features

    @best_features.setter
    def best_features(self, value: NpArray) -> None:
        self._best_features = value

    @property
    def best_params(self) -> dict:
        return self._best_params

    @best_params.setter
    def best_params(self, value: dict) -> None:
        self._best_params = value

    @property
    def initial_params(self) -> dict:
        return self._initial_params

    @initial_params.setter
    def initial_params(self, value: dict) -> None:
        self._initial_params = value

    @property
    def estimator(self) -> Any:
        return self._clf

    @estimator.setter
    def estimator(self, value: Any) -> None:
        self._clf = value

    @abstractmethod
    def score_model(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        pass


class SimpleModel(MachineLearningModel):

    def score_model(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        self.best_params = self.initial_params  # they are the same in a simple model
        # set clf params. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_params)
        self.best_features = x.columns.values  # get features as numpy data
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class OnlyFeatureSelectionModel(MachineLearningModel):

    def score_model(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        self.best_params = self.initial_params  # they are the same in a only feature selection model
        # set clf params. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_params)
        # get best features
        best_features_dataframe = self.feature_selector.select_features(x, y, clone(self.estimator))
        self.best_features = best_features_dataframe.columns.values  # get features as numpy data
        # x now has only the best features
        x = x[self.best_features]
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class OnlyParameterSearchModel(MachineLearningModel):

    def score_model(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        # transform initial params grid into a simple dict which is best_params
        self.best_params = self.parameter_selector.search_parameters(x, y, self.initial_params, 10,
                                                                     clone(self.estimator))
        self.best_features = x.columns.values  # get features as numpy data
        # set clf params from the previous search. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_params)
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class FeatureAndParameterSearch(MachineLearningModel):

    def score_model(self, df: DataFrame, score_type: str, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        # transform best params grid into a simple dict
        self.best_params = self.parameter_selector.search_parameters(x, y, self.initial_params, 10,
                                                                     clone(self.estimator))
        # set clf params from the previous search. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_params)
        # get best features
        best_features_dataframe = self.feature_selector.select_features(x, y, clone(self.estimator))
        self.best_features = best_features_dataframe.columns.values  # get features as numpy data
        # x now has only the best features
        x = x[self.best_features]
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, 10)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, 10)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class SBSModelCreator:
    __instance = None
    _types: dict = {"SM": SimpleModel(), "FSM": OnlyFeatureSelectionModel(),
                    "PSM": OnlyParameterSearchModel(), "AM": FeatureAndParameterSearch()}

    @staticmethod
    def get_instance() -> "SBSModelCreator":
        """Static access method."""
        if SBSModelCreator.__instance is None:
            SBSModelCreator()
        return SBSModelCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if SBSModelCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            SBSModelCreator.__instance = self

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
