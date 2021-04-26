from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from resources.backend_scripts.is_data import DataEnsurer

NpArray = np.ndarray
DataFrame = pd.DataFrame


class DataSplitter(ABC):

    @abstractmethod
    def train_test_split_data(self, x: DataFrame, y: NpArray, size: float) -> tuple:
        pass

    @abstractmethod
    def split_data_into_x_and_y(self, df: DataFrame, ravel: bool) -> tuple:
        pass


class NormalSplitter(DataSplitter):

    # interface method implementation
    def train_test_split_data(self, x: DataFrame, y: NpArray, size: float) -> tuple:
        # return x_train, x_test, y_train, y_test using scikitlearn train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
        temp_answer = x_train, x_test, y_train, y_test
        return temp_answer

    # interface method implementation
    def split_data_into_x_and_y(self, df: DataFrame, ravel: bool) -> tuple:
        # fix y as pd.DataFrame. By default it is a pd.Series
        y = df[df.columns[-1]].to_frame()
        x = df.drop([df.columns[-1]], axis=1)
        # it should returns a pd.DataFrame and a np.array
        if ravel:
            return x, y.values.ravel()
        return x, y


class SplitterReturner:

    @staticmethod
    def train_and_test_split(x: DataFrame, y: NpArray, size: float) -> tuple:
        data_splitter: DataSplitter = NormalSplitter()
        # if size is between valid range then return x and y train-test data
        if 0.0 < size < 1.0:
            tuple_answer = data_splitter.train_test_split_data(x, y, size)
            return tuple_answer
        raise ValueError("Size variable should be between 0.0 and 1.0 without them")

    @staticmethod
    def split_x_y_from_df(df: DataFrame, ravel_data=True) -> tuple:
        data_splitter: DataSplitter = NormalSplitter()
        if DataEnsurer.validate_pd_data(df):
            tuple_answer = data_splitter.split_data_into_x_and_y(df, ravel=ravel_data)
            return tuple_answer
        raise TypeError("The dataframe does no have enough samples or features to split them into x and y")
