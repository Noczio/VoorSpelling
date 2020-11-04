from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import pandas as pd
from sklearn.metrics import accuracy_score
from supervised import AutoML

from jsonInfo.random_generator import Randomizer
from split_data import SplitterReturner, NormalSplitter

T = TypeVar("T")
DataFrame = pd.DataFrame


class AutoMachineLearning(ABC, Generic[T]):

    def __init__(self, n_folds_validation: int, shuffle_data: bool, max_rand: int) -> None:
        # initialize _random_state, _n_folds_validation and _shuffle_data.
        if n_folds_validation < 3 or n_folds_validation > 10:
            raise ValueError("Number of folds is not greater than three and lesser to ten")
        elif max_rand < 0:
            raise ValueError("Random number must be a positive integer from zero to infinity")
        else:
            self._random_state: int = Randomizer.get_random_number_range_int(0, max_rand, 1)
            self._n_folds_validation: int = n_folds_validation
            self._shuffle_data: bool = shuffle_data

    @abstractmethod
    def fit_model(self, x_train: DataFrame, y_train: DataFrame) -> None:
        pass

    @abstractmethod
    def predict_model(self, x_test: DataFrame) -> tuple:
        pass

    @abstractmethod
    def get_model(self) -> T:
        pass


class JarAutoML(AutoMachineLearning[AutoML]):

    def __init__(self, n_folds_validation: int, shuffle_data: bool, max_rand: int) -> None:
        super().__init__(n_folds_validation, shuffle_data, max_rand)
        # initialize _clf as AutoMl type
        self._clf: AutoML = AutoML(
            mode="Compete",
            explain_level=0,
            random_state=self._random_state,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": self._n_folds_validation,
                "shuffle": self._shuffle_data
            })

    # abstract class method implementation
    def fit_model(self, x_train: DataFrame, y_train: DataFrame) -> None:
        # clf fit method
        self._clf.fit(x_train, y_train)

    # abstract class method implementation
    def predict_model(self, x_test: DataFrame) -> tuple:
        # clf predict. Returns prediction as tuple
        prediction_tuple = tuple(self._clf.predict(x_test))
        return prediction_tuple

    # abstract class method implementation
    def get_model(self) -> AutoML:
        model = self._clf
        return model


class AutoExecutioner:

    def __init__(self, auto_ml: AutoMachineLearning) -> None:
        # uses AutoMachineLearning and SplitterReturner(NormalSplitter())
        self._auto_ml = auto_ml
        self._splitter_returner = SplitterReturner(NormalSplitter())

    def get_model(self) -> str:
        model = self._auto_ml.get_model()
        return str(model)

    def train_model(self, df: DataFrame, size: float = 0.0) -> None:
        x, y = self._splitter_returner.split_x_y_from_df(df)
        if size == 0.0:
            self._auto_ml.fit_model(x, y)
        elif 0.0 < size < 1.0:
            x_train, _, y_train, _ = self._splitter_returner.train_and_test_split(x, y, size)
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
