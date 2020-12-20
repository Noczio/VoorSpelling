import pandas as pd
import numpy as np
from typing import Any

from feature_selection import FeatureSelection
from parameter_search import ParameterSearch

DataFrame = pd.DataFrame
NpArray = np.ndarray


class GlobalVariables:

    _df: DataFrame = pd.DataFrame()
    _fs: bool = False
    _ps: bool = False
    _fsm: FeatureSelection = None
    _psm: ParameterSearch = None
    _clf: Any = None
    __instance = None

    @staticmethod
    def get_instance() -> "GlobalVariables":
        """Static access method."""
        if GlobalVariables.__instance is None:
            GlobalVariables()
        return GlobalVariables.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if GlobalVariables.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            GlobalVariables.__instance = self

    @property
    def data_frame(self) -> DataFrame:
        return self._df

    @data_frame.setter
    def data_frame(self, value: DataFrame) -> None:
        self._df = value

    @property
    def uses_feature_selection(self) -> bool:
        return self._fs

    @uses_feature_selection.setter
    def uses_feature_selection(self, value: bool) -> None:
        self._fs = value

    @property
    def uses_parameter_search(self) -> bool:
        return self._ps

    @uses_parameter_search.setter
    def uses_parameter_search(self, value: bool) -> None:
        self._ps = value

    @property
    def feature_selection_method(self) -> FeatureSelection:
        return self._fsm

    @feature_selection_method.setter
    def feature_selection_method(self, value: FeatureSelection) -> None:
        self._fsm = value

    @property
    def parameter_search_method(self) -> ParameterSearch:
        return self._psm

    @parameter_search_method.setter
    def parameter_search_method(self, value: ParameterSearch) -> None:
        self._psm = value

    @property
    def estimator(self) -> Any:
        return self._clf

    @estimator.setter
    def estimator(self, value: Any) -> None:
        self._clf = value
