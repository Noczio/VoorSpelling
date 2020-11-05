from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.model_selection import cross_validate

DataFrame = pd.DataFrame


class FeatureSelection(ABC):

    @abstractmethod
    def select_features(self, x: DataFrame, y: DataFrame, model, **kwargs) -> tuple:
        pass


class CVModelScore(ABC):

    @abstractmethod
    def get_score(self, x: DataFrame, y: DataFrame, model, score_type: str,
                  n_folds_validation: int) -> Union[float, int]:
        pass


class CVScore(CVModelScore):
    _available_scores = ("accuracy", "f1", "roc_auc", "mutual_info_score", "r2", "precision")

    def get_score(self, x: DataFrame, y: DataFrame, model, score_type: str = "roc_auc",
                  n_folds_validation: int = 3) -> Union[float, int]:
        if n_folds_validation < 3 or n_folds_validation > 10:
            raise ValueError("Number of folds is not greater than three and lesser to ten")
        if score_type in self._available_scores:
            cv_results = cross_validate(model, x, y, cv=n_folds_validation, scoring=score_type)
            avg = cv_results['test_score'].mean()
            return avg
        raise ValueError("Parameter score_type is not one of _available_scores")


class OwnFeatureSelection(FeatureSelection):

    def __init__(self):
        self._cv_score = CVScore()

    def _first_iteration(self, x: DataFrame, y: DataFrame, model) -> tuple:
        score_lst = []
        for i in range(len(x.columns)):
            k = x.columns[i]
            temp_x = x[[k]]
            score = self._cv_score.get_score(temp_x, y, model)
            score_lst.append(score)

        max_score = max(score_lst)  # best score
        max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]  # index with best score
        top_score_index = max_score_index[0]
        new_feature = x.columns[top_score_index]
        new_best_x = x[new_feature]
        new_x = x.drop(new_feature, axis=1)

        return new_best_x, new_x, max_score

    def _else_iteration(self, best_x: DataFrame, x: DataFrame, y: DataFrame, model, actual_score: float):
        new_x_length = len(x.columns)
        if new_x_length > 0:
            score_lst = []
            for i in range(new_x_length):
                k = x.columns[i]
                temp_x = x[[k]]
                temp_new_x = pd.concat([best_x, temp_x], axis=1, ignore_index=True)
                score = self._cv_score.get_score(temp_new_x, y, model)
                score_lst.append(score)

            max_score = max(score_lst)  # best score

            if max_score < actual_score:
                return best_x, actual_score  # break condition, recursive function

            max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]  # index with best score
            top_score_index = max_score_index[0]

            new_feature = x.columns[top_score_index]
            temp_x = x[new_feature]
            new_best_x = pd.concat([best_x, temp_x], axis=1)
            new_x = x.drop(new_feature, axis=1)

            return self._else_iteration(new_best_x, new_x, y, model, max_score)

        return best_x, actual_score

    def select_features(self, x: DataFrame, y: DataFrame, model, **kwargs) -> tuple:
        f_best_x, new_x, f_score = self._first_iteration(x, y, model)
        best_x, best_score = self._else_iteration(f_best_x, new_x, y, model, f_score)

        return best_x, best_score
