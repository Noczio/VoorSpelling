from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import numpy as np

from is_data import DataEnsurer
from score import CVScore, CVModelScore

NpArray = np.ndarray
DataFrame = pd.DataFrame


class FeatureSelection(ABC):
    _cv_score: CVModelScore = CVScore()

    @abstractmethod
    def select_features(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                        n_folds_validation: int) -> DataFrame:
        pass


class BackwardsFeatureSelection(FeatureSelection):

    def __init__(self):
        self._initial_score: float = 0.0
        self._initial_x: DataFrame = DataFrame()

    def _iteration(self, x: DataFrame, y: NpArray, model: Any, actual_score: float, score_type: str,
                   n_folds_validation: int) -> tuple:
        # first check x len. this variable will become smaller and smaller over time
        new_x_length = len(x.columns)
        # if there are columns in the x dataframe then do the following process
        if new_x_length > 1:
            score_lst = []  # empty list to store score values
            # iterate over all features from x dataframe
            for i in range(new_x_length):
                # drop feature by index and store the results in a temp variable
                temp_col_name = x.columns[i]
                temp_x = x.drop([temp_col_name], axis=1)
                # get its score in a cv and append that value to the score_lst
                score = self._cv_score.get_score(temp_x, y, model, score_type, n_folds_validation)
                score_lst.append(score)
            # get the max score from the score_lst
            max_score = max(score_lst)
            # check if this iteration was worth it. If not then break the recursive method
            if actual_score > max_score:
                return x, actual_score
            # get the index of all score with max score
            max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]
            # get the top score from the max score index list
            if DataEnsurer.validate_py_data(max_score_index, list):
                top_score_index = max_score_index[0]
            else:
                top_score_index = max_score_index
            # drop the feature where the best score is. Without that feature the model improves
            temp_col_name = x.columns[top_score_index]
            new_best_x = x.drop([temp_col_name], axis=1)
            # finally return the best feature dataframe, the new x without that feature and the max score of this
            # iteration
            return self._iteration(new_best_x, y, model, max_score, score_type, n_folds_validation)

        # x dataframe is now empty, return the initial x dataframe and its score
        # this is bad scenario, because it iterated all features and there was not an improvement
        return self._initial_x, self._initial_score

    def select_features(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                        n_folds_validation: int) -> DataFrame:
        self._initial_x = x
        _, initial_y_shape = x.shape  # original column len for evaluation
        if initial_y_shape > 1:
            initial_score = self._cv_score.get_score(x, y, model, score_type,
                                                     n_folds_validation)
            # initial score with all features
            self._initial_score = initial_score
            # call recursive function and then return best x
            best_x, best_score = self._iteration(x, y, model, initial_score, score_type, n_folds_validation)
            # data might be len 1 or wrong
            if DataEnsurer.validate_py_data(best_x, pd.Series):
                return best_x.to_frame()
            elif DataEnsurer.validate_py_data(best_x, DataFrame):
                return best_x
            else:
                raise TypeError("Output is not a dataframe")
        else:
            return x


class ForwardFeatureSelection(FeatureSelection):

    def _first_iteration(self, x: DataFrame, y: NpArray, model: Any, score_type: str, n_folds_validation: int) -> tuple:
        score_lst = []  # empty list to store score values
        # iterate over all features
        for i in range(len(x.columns)):
            # in each iteration get the column
            temp_col_name = x.columns[i]
            # create a temp dataframe with the selected column
            temp_x = x[[temp_col_name]]
            # get its score in a cv and append that values to the score_lst
            score = self._cv_score.get_score(temp_x, y, model, score_type, n_folds_validation)
            score_lst.append(score)

        # get the max score from the score_lst
        max_score = max(score_lst)
        # get the index of all score with max score
        max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]
        # get the top score from the max score index list
        if DataEnsurer.validate_py_data(max_score_index, list):
            top_score_index = max_score_index[0]
        else:
            top_score_index = max_score_index
        # get feature name using top_score_index from original x dataframe
        new_feature = x.columns[top_score_index]
        # create a new dataframe with the winning feature
        new_best_x = x[new_feature]
        # drop the winning feature from the original dataframe and then store it into a new variable
        new_x = x.drop(new_feature, axis=1)
        # finally return the best feature dataframe, the new x without that feature and the max score of this iteration
        return new_best_x, new_x, max_score

    def _else_iteration(self, best_x: DataFrame, x: DataFrame, y: NpArray, model: Any, actual_score: float,
                        score_type: str, n_folds_validation: int) -> tuple:
        # first check x len. this variable will become smaller and smaller over time
        new_x_length = len(x.columns)
        # if there are columns in the x dataframe then do the following process
        if new_x_length > 0:
            score_lst = []  # empty list to store score values
            # iterate over all features from x dataframe
            for i in range(new_x_length):
                # in each iteration get the column
                temp_col_name = x.columns[i]
                # create a temp dataframe with the selected column
                temp_x = x[[temp_col_name]]
                temp_new_x = pd.concat([best_x, temp_x], axis=1, ignore_index=True)
                # get its score in a cv and append that values to the score_lst
                score = self._cv_score.get_score(temp_new_x, y, model, score_type, n_folds_validation)
                score_lst.append(score)

            # get the max score from the score_lst once the for loop has ended
            max_score = max(score_lst)  # best score
            # check if this iteration was worth it. If not then break the recursive method
            if max_score < actual_score:
                return best_x, actual_score

            # get the index of all score with max score
            max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]
            # get the top score from the max score index list
            if DataEnsurer.validate_py_data(max_score_index, list):
                top_score_index = max_score_index[0]
            else:
                top_score_index = max_score_index
            # get feature name using top_score_index from original x dataframe
            new_feature = x.columns[top_score_index]
            # create a new dataframe with the winning feature
            temp_x = x[new_feature]
            # create a variable to store the last best dataframe with this new best feature group
            new_best_x = pd.concat([best_x, temp_x], axis=1)
            # drop the winning feature from the x dataframe and then store it into a new variable
            new_x = x.drop(new_feature, axis=1)
            # call the recursive function all over again until the condition is met
            return self._else_iteration(new_best_x, new_x, y, model, max_score, score_type, n_folds_validation)

        # x dataframe is now empty, return best x dataframe and its score
        # this is bad scenario, because it iterated all features and there was not an improvement
        return best_x, actual_score

    def select_features(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                        n_folds_validation: int) -> DataFrame:
        _, initial_y_shape = x.shape  # original column len for evaluation
        # if x only has 1 column then return original dataframe
        if initial_y_shape == 1:
            return x
        else:
            # else if x has more than 1 column
            f_best_x, new_x, f_score = self._first_iteration(x, y, model, score_type, n_folds_validation)
            # call recursive function and then return best x
            best_x, best_score = self._else_iteration(f_best_x, new_x, y, model, f_score, score_type,
                                                      n_folds_validation)
            # data might be len 1 or wrong
            if DataEnsurer.validate_py_data(best_x, pd.Series):
                return best_x.to_frame()
            elif DataEnsurer.validate_py_data(best_x, DataFrame):
                return best_x
            else:
                raise TypeError("Output is not a dataframe")


class FeatureSelectorCreator:
    __instance = None
    _types: dict = {"FFS": ForwardFeatureSelection(), "BFS": BackwardsFeatureSelection()}

    @staticmethod
    def get_instance() -> "FeatureSelectorCreator":
        """Static access method."""
        if FeatureSelectorCreator.__instance is None:
            FeatureSelectorCreator()
        return FeatureSelectorCreator.__instance

    def __init__(self) -> None:
        """Virtually private constructor."""
        if FeatureSelectorCreator.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            FeatureSelectorCreator.__instance = self

    def create_feature_selector(self, selection_type: str) -> FeatureSelection:
        # transform param to capital letters and then replace white spaces
        key = selection_type.upper().replace(" ", "")
        if key in self._types.keys():
            feature_selection_type = self._types[key]
            return feature_selection_type
        raise KeyError("Feature selection key value is wrong. It should be: FFS or BFS")

    def get_available_types(self) -> tuple:
        available_types = [k for k in self._types.keys()]
        types = tuple(available_types)
        return types
