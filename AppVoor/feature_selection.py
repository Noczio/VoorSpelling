from abc import ABC, abstractmethod

import pandas as pd

from score import CVModelScore, CVScore

DataFrame = pd.DataFrame


class FeatureSelection(ABC):

    def __init__(self, score_type: str, n_folds_validation: int, cv_score: CVModelScore) -> None:
        self._score_type = score_type
        self._n_folds_validation = n_folds_validation
        self._cv_score = cv_score

    @abstractmethod
    def select_features(self, x: DataFrame, y: DataFrame, model) -> tuple:
        pass


class OwnFeatureSelection(FeatureSelection):

    def _first_iteration(self, x: DataFrame, y: DataFrame, model) -> tuple:
        score_lst = []
        for i in range(len(x.columns)):
            k = x.columns[i]
            temp_x = x[[k]]
            score = self._cv_score.get_score(temp_x, y, model, self._score_type, self._n_folds_validation)
            score_lst.append(score)

        max_score = max(score_lst)  # best score
        max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]  # index with best score
        top_score_index = max_score_index[0]
        new_feature = x.columns[top_score_index]
        new_best_x = x[new_feature]
        new_x = x.drop(new_feature, axis=1)

        return new_best_x, new_x, max_score

    def _else_iteration(self, best_x: DataFrame, x: DataFrame, y: DataFrame, model, actual_score: float) -> tuple:
        new_x_length = len(x.columns)
        if new_x_length > 0:
            score_lst = []
            for i in range(new_x_length):
                k = x.columns[i]
                temp_x = x[[k]]
                temp_new_x = pd.concat([best_x, temp_x], axis=1, ignore_index=True)
                score = self._cv_score.get_score(temp_new_x, y, model, self._score_type, self._n_folds_validation)
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

    def select_features(self, x: DataFrame, y: DataFrame, model) -> tuple:
        _, initial_y_shape = x.shape
        # if x only has 1 column then, return dataframe with its score
        if initial_y_shape == 1:
            score = self._cv_score.get_score(x, y, model, self._score_type, self._n_folds_validation)
            return x, score
        # else if x has more than 1 column
        f_best_x, new_x, f_score = self._first_iteration(x, y, model)
        best_x, best_score = self._else_iteration(f_best_x, new_x, y, model, f_score)

        return best_x, best_score


class SFMFeatureSelection(FeatureSelection):

    def select_features(self, x: DataFrame, y: DataFrame, model) -> tuple:
        pass
