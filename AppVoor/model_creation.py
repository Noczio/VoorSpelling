from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from feature_selection import FeatureSelection
from is_data import DataEnsurer
from parameter_search import ParameterSearch
from score import CVScore, CVModelScore
from split_data import SplitterReturner
from switcher import Switch

DataFrame = pd.DataFrame
NpArray = np.ndarray


class SBSMachineLearning(ABC):
    _feature_selector: FeatureSelection = None
    _parameter_selector: ParameterSearch = None
    _best_features: NpArray = None
    _best_params: dict = None
    _initial_params: dict = None
    _clf: Any = None
    _cv_score: CVModelScore = CVScore()

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
    def best_parameters(self) -> dict:
        return self._best_params

    @best_parameters.setter
    def best_parameters(self, value: dict) -> None:
        self._best_params = value

    @property
    def initial_parameters(self) -> dict:
        return self._initial_params

    @initial_parameters.setter
    def initial_parameters(self, value: dict) -> None:
        self._initial_params = value

    @property
    def estimator(self) -> Any:
        return self._clf

    @estimator.setter
    def estimator(self, value: Any) -> None:
        self._clf = value

    @abstractmethod
    def score_model(self, df: DataFrame, score_type: str, n_folds_validation: int, size: float = 0.0) -> float:
        pass


class SimpleSBS(SBSMachineLearning):

    def score_model(self, df: DataFrame, score_type: str, n_folds_validation: int, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        self.best_parameters = self.initial_parameters  # they are the same in a simple model
        # set clf params. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        self.best_features = x.columns.values  # get features as numpy data
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, n_folds_validation)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, n_folds_validation)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class OnlyFeatureSelectionSBS(SBSMachineLearning):

    def score_model(self, df: DataFrame, score_type: str, n_folds_validation: int, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        self.best_parameters = self.initial_parameters  # they are the same in a only feature selection model
        # set clf params. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        # get best features
        best_features_dataframe = self.feature_selector.select_features(x, y, clone(self.estimator), score_type,
                                                                        n_folds_validation)
        self.best_features = best_features_dataframe.columns.values  # get features as numpy data
        # x now has only the best features
        x = x[self.best_features]
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, n_folds_validation)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, n_folds_validation)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class OnlyParameterSearchSBS(SBSMachineLearning):

    def score_model(self, df: DataFrame, score_type: str, n_folds_validation: int, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        # transform initial params grid into a simple dict which is best_params
        self.best_parameters = self.parameter_selector.search_parameters(x, y, self.initial_parameters,
                                                                         n_folds_validation,
                                                                         clone(self.estimator))
        self.best_features = x.columns.values  # get features as numpy data
        # set clf params from the previous search. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, n_folds_validation)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, n_folds_validation)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class FeatureAndParameterSearchSBS(SBSMachineLearning):

    def score_model(self, df: DataFrame, score_type: str, n_folds_validation: int, size: float = 0.0) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(df)
        # transform best params grid into a simple dict
        self.best_parameters = self.parameter_selector.search_parameters(x, y, self.initial_parameters,
                                                                         n_folds_validation,
                                                                         clone(self.estimator))
        # set clf params from the previous search. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        # get best features
        best_features_dataframe = self.feature_selector.select_features(x, y, clone(self.estimator), score_type,
                                                                        n_folds_validation)
        self.best_features = best_features_dataframe.columns.values  # get features as numpy data
        # x now has only the best features
        x = x[self.best_features]
        # return the cv score
        if size == 0.0:
            score = self._cv_score.get_score(x, y, self.estimator, score_type, n_folds_validation)
            return score
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            score = self._cv_score.get_score(x_train, y_train, self.estimator, score_type, n_folds_validation)
            return score
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")


class ModelPossibilities(Switch):

    @staticmethod
    def SM() -> SBSMachineLearning:
        return SimpleSBS()

    @staticmethod
    def FSM() -> SBSMachineLearning:
        return OnlyFeatureSelectionSBS()

    @staticmethod
    def PSM() -> SBSMachineLearning:
        return OnlyParameterSearchSBS()

    @staticmethod
    def AM() -> SBSMachineLearning:
        return FeatureAndParameterSearchSBS()

    @staticmethod
    def Simple() -> SBSMachineLearning:
        return SimpleSBS()

    @staticmethod
    def OnlyFeatureSelection() -> SBSMachineLearning:
        return OnlyFeatureSelectionSBS()

    @staticmethod
    def OnlyParameterSearch() -> SBSMachineLearning:
        return OnlyParameterSearchSBS()

    @staticmethod
    def FeatureAndParameterSearch() -> SBSMachineLearning:
        return FeatureAndParameterSearchSBS()


class SBSModelCreator:
    __instance = None

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

    def create_model(self, feature_selection: bool, parameter_search: bool) -> SBSMachineLearning:
        if DataEnsurer.validate_py_data(feature_selection, bool) and DataEnsurer.validate_py_data(parameter_search,
                                                                                                  bool):
            if not feature_selection and not parameter_search:
                simple_model = ModelPossibilities.case("SM")
                return simple_model
            elif feature_selection and not parameter_search:
                only_feature_selection_model = ModelPossibilities.case("FSM")
                return only_feature_selection_model
            elif not feature_selection and parameter_search:
                only_parameter_search_model = ModelPossibilities.case("PSM")
                return only_parameter_search_model
            else:
                all_model = ModelPossibilities.case("AM")
                return all_model
        raise TypeError("Both parameters should be Boolean type")

    def get_available_types(self) -> tuple:
        available_types = [func for func in dir(ModelPossibilities)
                           if callable(getattr(ModelPossibilities, func)) and not
                           (func.startswith("__") or func is "case")]
        return tuple(available_types)
