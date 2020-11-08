from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import numpy as np

from jsonInfo.json_to_data import JSONMessage
from score import CVModelScore, CVScore
from sklearn.feature_selection import SelectFromModel, RFE


NpArray = np.ndarray
DataFrame = pd.DataFrame


class FeatureSelection(ABC):

    @abstractmethod
    def select_features(self, x: DataFrame, y: NpArray, model: Any) -> DataFrame:
        pass


class SFMFeatureSelection(FeatureSelection):

    def select_features(self, x: DataFrame, y: NpArray, model: Any) -> DataFrame:
        clf = model.fit(x, y)
        sfm = SelectFromModel(clf, prefit=True)
        _ = sfm.transform(x)
        features = x.columns[sfm.get_support()]
        transformed_x = x[features]
        return transformed_x


class RFEFeatureSelection(FeatureSelection):

    def select_features(self, x: DataFrame, y: NpArray, model: Any) -> DataFrame:
        clf = model
        rfe = RFE(clf)
        rfe.fit(x, y)
        _ = rfe.transform(x)
        features = x.columns[rfe.get_support()]
        transformed_x = x[features]
        return transformed_x
