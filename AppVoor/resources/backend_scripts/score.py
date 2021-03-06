from abc import abstractmethod, ABC
from typing import Any

from sklearn.model_selection import cross_validate

import pandas as pd
import numpy as np

NpArray = np.ndarray
DataFrame = pd.DataFrame


class CVModelScore(ABC):

    @abstractmethod
    def get_score(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                  n_folds_validation: int) -> float:
        pass

    @abstractmethod
    def get_available_types(self) -> tuple:
        pass


class CVScore(CVModelScore):
    _types = ("roc_auc", "accuracy",
              "r2", "neg_mean_squared_error", "explained_variance",
              "mutual_info_score", "completeness_score")

    def get_score(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                  n_folds_validation: int) -> float:

        if n_folds_validation < 3 or n_folds_validation > 10 and score_type in self._types:
            raise ValueError("Number of folds is not greater than three and lesser to ten")
        elif score_type not in self._types and (3 <= n_folds_validation <= 10):
            raise ValueError("Parameter score_type is not one of available_scores")
        elif score_type not in self._types and n_folds_validation < 3 or n_folds_validation > 10:
            raise ValueError("Parameters n_folds_validation and score_type have wrong values")
        else:
            cv_results = cross_validate(model, x, y, cv=n_folds_validation, scoring=score_type)
            avg = cv_results['test_score'].mean()
            return avg

    def get_available_types(self) -> tuple:
        return self._types
