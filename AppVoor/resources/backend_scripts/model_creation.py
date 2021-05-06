from abc import abstractmethod, ABC
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from resources.backend_scripts.feature_selection import FeatureSelection
from resources.backend_scripts.is_data import DataEnsurer
from resources.backend_scripts.parameter_search import ParameterSearch
from resources.backend_scripts.score import CVScore, CVModelScore
from resources.backend_scripts.split_data import SplitterReturner
from resources.backend_scripts.switcher import Switch

DataFrame = pd.DataFrame
NpArray = np.ndarray


class SBSMachineLearning(ABC):
    _data_frame = pd.DataFrame()
    _feature_selector: FeatureSelection = None
    _parameter_selector: ParameterSearch = None
    _best_features: NpArray = None
    _best_params: dict = None
    _initial_params: dict = None
    _clf: Any = None
    _cv_score: CVModelScore = CVScore()

    @property
    def data_frame(self) -> DataFrame:
        return self._data_frame

    @data_frame.setter
    def data_frame(self, value: DataFrame) -> None:
        self._data_frame = value

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
    def score_model(self, score_type: str, n_folds_validation: int) -> float:
        pass


class SimpleSBS(SBSMachineLearning):

    def score_model(self, score_type: str, n_folds_validation: int) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(self.data_frame)
        self.best_parameters = self.initial_parameters  # they are the same in a simple model
        # set clf params. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        self.best_features = x.columns.values  # get features as numpy data
        # return the cv score
        score = self._cv_score.get_score(x, y, clone(self.estimator), score_type, n_folds_validation)
        self.estimator.fit(x, y)
        return score


class OnlyFeatureSelectionSBS(SBSMachineLearning):

    def score_model(self, score_type: str, n_folds_validation: int) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(self.data_frame, ravel_data=False)
        self.best_parameters = self.initial_parameters  # they are the same in a only feature selection model
        # set clf params. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        # get best features
        best_features_dataframe, score = self.feature_selector.select_features(x, y, clone(self.estimator),
                                                                               score_type, n_folds_validation)
        self.best_features = best_features_dataframe.columns.values  # get features as numpy data
        self.data_frame = pd.concat([best_features_dataframe, y], axis=1)
        self.estimator.fit(best_features_dataframe, y)
        return score


class OnlyParameterSearchSBS(SBSMachineLearning):

    def score_model(self, score_type: str, n_folds_validation: int) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(self.data_frame)
        # transform initial params grid into a simple dict which is best_params
        self.best_parameters, score = self.parameter_selector.search_parameters(x, y, self.initial_parameters,
                                                                                n_folds_validation,
                                                                                clone(self.estimator),
                                                                                score_type)
        self.best_features = x.columns.values  # get features as numpy data
        # set clf params from the previous search. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        self.estimator.fit(x, y)
        return score


class FeatureAndParameterSearchSBS(SBSMachineLearning):

    def score_model(self, score_type: str, n_folds_validation: int) -> float:
        # get x and y from df
        x, y = SplitterReturner.split_x_y_from_df(self.data_frame, ravel_data=False)
        # transform best params grid into a simple dict
        self.best_parameters, _ = self.parameter_selector.search_parameters(x, y, self.initial_parameters,
                                                                            n_folds_validation,
                                                                            clone(self.estimator),
                                                                            score_type)
        # set clf params from the previous search. ** because it accepts key-value one by one, not a big dictionary
        self.estimator.set_params(**self.best_parameters)
        # get best features
        best_features_dataframe, score = self.feature_selector.select_features(x, y, clone(self.estimator),
                                                                               score_type, n_folds_validation)
        self.best_features = best_features_dataframe.columns.values  # get features as numpy data
        self.data_frame = pd.concat([best_features_dataframe, y], axis=1)
        self.estimator.fit(best_features_dataframe, y)
        return score


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

    @staticmethod
    def create_model(feature_selection: bool, parameter_search: bool) -> SBSMachineLearning:
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

    @staticmethod
    def get_available_types() -> tuple:
        available_types = [func for func in dir(ModelPossibilities)
                           if callable(getattr(ModelPossibilities, func)) and not
                           (func.startswith("__") or func is "case")]
        return tuple(available_types)
