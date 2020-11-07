from abc import abstractmethod
from typing import Union, Any

from sklearn.model_selection import cross_validate

from jsonInfo.json_to_data import JSONMessage
from jsonInfo.metrics import CVMetrics

import pandas as pd

DataFrame = pd.DataFrame


class CVModelScore(ABC):

    def __init__(self, file_path: str, data_type: Any):
        # uses a JSONMessage implementation
        json_path = file_path
        json_data_type = data_type
        cv_metrics: JSONMessage = CVMetrics(file_path=json_path, data_type=json_data_type)
        self._available_scores = cv_metrics.data

    @abstractmethod
    def get_score(self, x: DataFrame, y: DataFrame, model: Any, score_type: str,
                  n_folds_validation: int) -> Union[float, int]:
        pass


class CVScore(CVModelScore):

    def get_score(self, x: DataFrame, y: DataFrame, model: Any, score_type: str,
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
