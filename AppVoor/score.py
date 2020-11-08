from abc import abstractmethod, ABC
from typing import Union, Any

from sklearn.model_selection import cross_validate

from jsonInfo.json_to_data import JSONMessage

import pandas as pd
import numpy as np

NpArray = np.ndarray
DataFrame = pd.DataFrame


class CVModelScore(ABC):

    def __init__(self, cv_metrics: JSONMessage):
        # uses a JSONMessage implementation
        cv_metrics: JSONMessage = cv_metrics
        self._available_scores = cv_metrics.data

    @abstractmethod
    def get_score(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                  n_folds_validation: int) -> Union[float, int]:
        pass


class CVScore(CVModelScore):

    def get_score(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                  n_folds_validation: int) -> Union[float, int]:

        if n_folds_validation < 3 or n_folds_validation > 10:
            raise ValueError("Number of folds is not greater than three and lesser to ten")
        elif score_type not in self._available_scores:
            raise ValueError("Parameter score_type is not one of available_scores")
        elif score_type not in self._available_scores and n_folds_validation < 3 or n_folds_validation > 10:
            raise ValueError("Parameters n_folds_validation and score_type have wrong values")
        else:
            cv_results = cross_validate(model, x, y, cv=n_folds_validation, scoring=score_type)
            avg = cv_results['test_score'].mean()
            return avg
