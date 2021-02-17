from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from supervised import AutoML

from jsonInfo.random_generator import Randomizer
from backend_scripts.split_data import SplitterReturner

NpArray = np.ndarray
DataFrame = pd.DataFrame


class AutoMachineLearning(ABC):
    _estimator: Any = None

    def __init__(self, n_folds_validation: int, shuffle_data: bool, max_rand: int) -> None:
        # initialize _random_state, _n_folds_validation and _shuffle_data.
        if n_folds_validation < 3 or n_folds_validation > 10:
            raise ValueError("Number of folds is not greater than three and lesser to ten")
        elif max_rand < 0:
            raise ValueError("Random number must be a positive integer from zero to infinity")
        else:
            self._random_state: int = Randomizer.get_random_number_range_int(0, max_rand + 1, 1)
            self._n_folds_validation: int = n_folds_validation
            self._shuffle_data: bool = shuffle_data

    @abstractmethod
    def fit_model(self, x_train: DataFrame, y_train: NpArray) -> None:
        pass

    @abstractmethod
    def predict_model(self, x_test: DataFrame) -> tuple:
        pass

    @property
    def estimator(self) -> Any:
        return self._estimator

    @estimator.setter
    def estimator(self, value: Any) -> None:
        self._estimator = value


class JarAutoML(AutoMachineLearning):

    def __init__(self, n_folds_validation: int, shuffle_data: bool, max_rand: int) -> None:
        super().__init__(n_folds_validation, shuffle_data, max_rand)
        # initialize _clf as AutoMl type
        self.estimator = AutoML(
            mode="Compete",
            explain_level=0,
            random_state=self._random_state,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": self._n_folds_validation,
                "shuffle": self._shuffle_data
            })

    # abstract class method implementation
    def fit_model(self, x_train: DataFrame, y_train: NpArray) -> None:
        # clf fit method
        self.estimator.fit(x_train, y_train)

    # abstract class method implementation
    def predict_model(self, x_test: DataFrame) -> tuple:
        # clf predict. Returns prediction as tuple
        prediction_tuple = tuple(self.estimator.predict(x_test))
        return prediction_tuple


class AutoExecutioner:

    def __init__(self, auto_ml: AutoMachineLearning) -> None:
        self._auto_ml = auto_ml

    def __str__(self):
        return self.get_model()

    def get_model(self) -> str:
        model = self._auto_ml.estimator
        return str(model)

    def train_model(self, df: DataFrame, size: float = 0.0) -> None:
        x, y = SplitterReturner.split_x_y_from_df(df)
        if size == 0.0:
            self._auto_ml.fit_model(x, y)
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = SplitterReturner.train_and_test_split(x, y, size)
            self._auto_ml.fit_model(x_train, y_train)
        else:
            raise ValueError("Size is neither 0.0 nor 0.0 < size < 1.0")

    def score_model(self, test: DataFrame, expected: tuple) -> float:
        # get the number of samples is test dataframe
        x_len, _ = test.shape
        # if x_len is not the same len as expected prediction then raise a ValueError
        if x_len == len(expected):
            prediction = self._auto_ml.predict_model(test)
            score = accuracy_score(expected, prediction)
            return score
        raise ValueError("Test samples is not the same size as expected prediction")
