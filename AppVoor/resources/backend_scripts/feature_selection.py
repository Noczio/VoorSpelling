from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from resources.backend_scripts.is_data import DataEnsurer
from resources.backend_scripts.score import CVScore, CVModelScore
from resources.backend_scripts.switcher import Switch

NpArray = np.ndarray
DataFrame = pd.DataFrame


class FeatureSelection(ABC):
    _cv_score: CVModelScore = CVScore()

    @abstractmethod
    def select_features(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                        n_folds_validation: int) -> tuple:
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
                print("Temp column name to drop:", temp_col_name)
                temp_x = x.drop([temp_col_name], axis=1)
                # get its score in a cv and append that value to the score_lst
                score = self._cv_score.get_score(temp_x, y, model, score_type, n_folds_validation)
                print("Score with last temp column dropped:", score)
                score_lst.append(score)
            # get the max score from the score_lst
            max_score = max(score_lst)
            print("Max score from this iteration:", max_score)
            # check if this iteration was worth it. If not then break the recursive method
            if is_greater_than(actual_score, max_score, score_type):
                print("There was not an improvement in this iteration")
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
            print("There was an improvement in this iteration")
            return self._iteration(new_best_x, y, model, max_score, score_type, n_folds_validation)

        # x dataframe is now empty, return the initial x dataframe and its score
        # this is bad scenario, because it iterated all features and there was not an improvement
        return self._initial_x, self._initial_score

    def select_features(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                        n_folds_validation: int) -> tuple:
        self._initial_x = x
        _, initial_y_shape = x.shape  # original column len for evaluation
        if initial_y_shape > 1:
            initial_score = self._cv_score.get_score(x, y, model, score_type,
                                                     n_folds_validation)
            # initial score with all features
            self._initial_score = initial_score
            # call recursive function and then return best x
            print("Feature selection process started")
            best_x, best_score = self._iteration(x, y, model, initial_score, score_type, n_folds_validation)
            # data might be len 1 or wrong
            print("Feature selection process finished")
            if DataEnsurer.validate_py_data(best_x, pd.Series):
                return best_x.to_frame(), best_score
            elif DataEnsurer.validate_py_data(best_x, DataFrame):
                return best_x, best_score
            else:
                raise TypeError("Output is not a dataframe")
        else:
            raise ValueError("Not enough columns to start feature selection")


class ForwardFeatureSelection(FeatureSelection):

    def _first_iteration(self, x: DataFrame, y: NpArray, model: Any, score_type: str, n_folds_validation: int) -> tuple:
        score_lst = []  # empty list to store score values
        # iterate over all features
        for i in range(len(x.columns)):
            # in each iteration get the column
            temp_col_name = x.columns[i]
            print("Temp column name: ", temp_col_name)
            # create a temp dataframe with the selected column
            temp_x = x[[temp_col_name]]
            # get its score in a cv and append that values to the score_lst
            score = self._cv_score.get_score(temp_x, y, model, score_type, n_folds_validation)
            print("Score with last temp column: ", score)
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
        print("X n of columns: ", new_x_length)
        # if there are columns in the x dataframe then do the following process
        if new_x_length > 0:
            score_lst = []  # empty list to store score values
            # iterate over all features from x dataframe
            for i in range(new_x_length):
                # in each iteration get the column
                temp_col_name = x.columns[i]
                print("Temp column name: ", temp_col_name)
                # create a temp dataframe with the selected column
                temp_x = x[[temp_col_name]]
                temp_new_x = pd.concat([best_x, temp_x], axis=1, ignore_index=True)
                # get its score in a cv and append that values to the score_lst
                score = self._cv_score.get_score(temp_new_x, y, model, score_type, n_folds_validation)
                print("Score with last temp column: ", score)
                score_lst.append(score)

            # get the max score from the score_lst once the for loop has ended
            max_score = max(score_lst)  # best score
            print("Max score from this iteration: ", max_score)
            # check if this iteration was worth it. If not then break the recursive method
            if is_fewer_than(max_score, actual_score, score_type):
                print("There was not an improvement in this iteration")
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
            print("There was an improvement in this iteration")
            return self._else_iteration(new_best_x, new_x, y, model, max_score, score_type, n_folds_validation)

        # x dataframe is now empty, return best x dataframe and its score
        # this is bad scenario, because it iterated all features and there was not an improvement
        return best_x, actual_score

    def select_features(self, x: DataFrame, y: NpArray, model: Any, score_type: str,
                        n_folds_validation: int) -> tuple:
        _, initial_y_shape = x.shape  # original column len for evaluation
        # if x only has 1 column then return original dataframe
        if initial_y_shape > 1:
            # else if x has more than 1 column
            print("Feature selection process started")
            f_best_x, new_x, f_score = self._first_iteration(x, y, model, score_type, n_folds_validation)
            # call recursive function and then return best x
            best_x, best_score = self._else_iteration(f_best_x, new_x, y, model, f_score, score_type,
                                                      n_folds_validation)
            # data might be len 1 or wrong
            print("Feature selection process finished")
            if DataEnsurer.validate_py_data(best_x, pd.Series):
                return best_x.to_frame(), best_score
            elif DataEnsurer.validate_py_data(best_x, DataFrame):
                return best_x, best_score
            else:
                raise TypeError("Output is not a dataframe")
        else:
            raise ValueError("Not enough columns to start feature selection")


class FeatureSelectionPossibilities(Switch):

    @staticmethod
    def FFS() -> FeatureSelection:
        return ForwardFeatureSelection()

    @staticmethod
    def BFS() -> FeatureSelection:
        return BackwardsFeatureSelection()

    @staticmethod
    def ForwardFeatureSelection() -> FeatureSelection:
        return ForwardFeatureSelection()

    @staticmethod
    def BackwardsFeatureSelection() -> FeatureSelection:
        return BackwardsFeatureSelection()


class FeatureSelectorCreator:

    @staticmethod
    def create_feature_selector(selection_type: str) -> FeatureSelection:
        try:
            feature_selection_name = selection_type.replace(" ", "")
            feature_selection_method = FeatureSelectionPossibilities.case(feature_selection_name)
            return feature_selection_method
        except():
            available_types = FeatureSelectorCreator.get_available_types()
            types_as_string = ", ".join(available_types)
            raise AttributeError(f"Parameter value is wrong. "
                                 f"It should be any of the following: {types_as_string}")

    @staticmethod
    def get_available_types() -> tuple:
        available_types = [func for func in dir(FeatureSelectionPossibilities)
                           if callable(getattr(FeatureSelectionPossibilities, func)) and not
                           (func.startswith("__") or func is "case")]
        return tuple(available_types)


# Patch when a score is not the bigger the better. The first is fewer than the second
def is_fewer_than(first: float, second: float, score_type: str):
    if score_type is "roc_auc" or "accuracy" or "mutual_info_score" or "completeness_score":
        return first < second
    return second < first


# Patch when a score is not the bigger the better. The first is greater than the second
def is_greater_than(first: float, second: float, score_type: str):
    if score_type is "roc_auc" or "accuracy" or "mutual_info_score" or "completeness_score":
        return first > second
    return second > first
