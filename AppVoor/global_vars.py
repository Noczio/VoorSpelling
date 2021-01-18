from typing import Any

import numpy as np
import pandas as pd

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
    _prd_type: str = ""
    _initial_value = {"data_frame": pd.DataFrame(),
                      "uses_feature_selection": False,
                      "uses_parameter_search": False,
                      "feature_selection_method": None,
                      "parameter_search_method": None,
                      "estimator": None,
                      "prediction_type": ""
                      }
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

    @property
    def prediction_type(self) -> Any:
        return self._prd_type

    @prediction_type.setter
    def prediction_type(self, value: str) -> None:
        self._prd_type = value

    @classmethod
    def reset(cls, *args, **kwargs):
        if len(kwargs) == 0 and len(args) == 0:
            cls.data_frame = cls._initial_value["data_frame"]
            cls.uses_feature_selection = cls._initial_value["uses_feature_selection"]
            cls.uses_parameter_search = cls._initial_value["uses_parameter_search"]
            cls.feature_selection_method = cls._initial_value["feature_selection_method"]
            cls.parameter_search_method = cls._initial_value["parameter_search_method"]
            cls.estimator = cls._initial_value["estimator"]
            cls.prediction_type = cls._initial_value["prediction_type"]
        elif len(kwargs) > 0 and len(args) == 0:
            for key, value in kwargs.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
        elif len(kwargs) == 0 and len(args) > 0:
            for key in args:
                if hasattr(cls, key):
                    setattr(cls, key, cls._initial_value[key])
        else:
            for key in args:
                if hasattr(cls, key):
                    setattr(cls, key, cls._initial_value[key])
            for key, value in kwargs.items():
                if hasattr(cls, key):
                    setattr(cls, key, value)
