from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE

from score import CVScore, CVModelScore

NpArray = np.ndarray
DataFrame = pd.DataFrame


class FeatureSelection(ABC):

    @abstractmethod
    def select_features(self, x: DataFrame, y: NpArray, model: Any) -> DataFrame:
        pass


class ForwardFeatureSelection(FeatureSelection):
    _cv_score: CVModelScore = CVScore()

    def _first_iteration(self, x: DataFrame, y: NpArray, model: Any) -> tuple:
        score_lst = []  # empty list to store score values
        # iterate over all features
        for i in range(len(x.columns)):
            # in each iteration get the column
            k = x.columns[i]
            # create a temp dataframe with the selected column
            temp_x = x[[k]]
            # get its score in a cv and append that values to the score_lst
            score = self._get_cv_score(temp_x, y, model)
            score_lst.append(score)

        # get the max score from the score_lst
        max_score = max(score_lst)
        # get the index of all score with max score
        max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]
        # get the top score from the max score index list
        top_score_index = max_score_index[0]
        # get feature name using top_score_index from original x dataframe
        new_feature = x.columns[top_score_index]
        # create a new dataframe with the winning feature
        new_best_x = x[new_feature]
        # drop the winning feature from the original dataframe and then store it into a new variable
        new_x = x.drop(new_feature, axis=1)
        # finally return the best feature dataframe, the new x without that feature and the max score of this iteration
        return new_best_x, new_x, max_score

    def _else_iteration(self, best_x: DataFrame, x: DataFrame, y: NpArray, model, actual_score: float) -> tuple:
        # first check x len. this variable will become smaller and smaller over time
        new_x_length = len(x.columns)
        # if there are columns in the x dataframe then do the following process
        if new_x_length > 0:
            score_lst = []  # empty list to store score values
            # iterate over all features from x dataframe
            for i in range(new_x_length):
                # in each iteration get the column
                k = x.columns[i]
                # create a temp dataframe with the selected column
                temp_x = x[[k]]
                temp_new_x = pd.concat([best_x, temp_x], axis=1, ignore_index=True)
                # get its score in a cv and append that values to the score_lst
                score = self._get_cv_score(temp_new_x, y, model)
                score_lst.append(score)

            # get the max score from the score_lst once the for loop has ended
            max_score = max(score_lst)  # best score
            # check if this iteration was worth it. If not then break the recursive method
            if max_score < actual_score:
                return best_x, actual_score

            # get the index of all score with max score
            max_score_index = [i for i, j in enumerate(score_lst) if j == max_score]
            # get the top score from the max score index list
            top_score_index = max_score_index[0]
            # get feature name using top_score_index from original x dataframe
            new_feature = x.columns[top_score_index]
            # create a new dataframe with the winning feature
            temp_x = x[new_feature]
            # create a variable to store the last best dataframe with this new best feature group
            new_best_x = pd.concat([best_x, temp_x], axis=1)
            # drop the winning feature from the x dataframe and then store it into a new variable
            new_x = x.drop(new_feature, axis=1)
            # call the recursive function all over again until the condition is met
            return self._else_iteration(new_best_x, new_x, y, model, max_score)

        # x dataframe is now empty, return best x dataframe and its score
        # this is bad scenario, because it iterated all features and there was not an improvement
        return best_x, actual_score

    def _get_cv_score(self, x: DataFrame, y: NpArray, model: Any) -> float:
        # get score using the object and the method parameters and the return it
        score = self._cv_score.get_score(x, y, model, "roc_auc", 10)
        return score

    def select_features(self, x: DataFrame, y: NpArray, model: Any) -> DataFrame:
        _, initial_y_shape = x.shape  # original column len for evaluation
        # if x only has 1 column then return original dataframe
        if initial_y_shape == 1:
            return x
        else:
            # else if x has more than 1 column
            f_best_x, new_x, f_score = self._first_iteration(x, y, model)
            # call recursive function and then return best x
            best_x, best_score = self._else_iteration(f_best_x, new_x, y, model, f_score)
            return best_x


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
