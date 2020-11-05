from abc import abstractmethod, ABC

import pandas as pd
from sklearn.model_selection import train_test_split

from is_data import DataEnsurer

DataFrame = pd.DataFrame


class DataSplitter(ABC):

    @abstractmethod
    def train_test_split_data(self, x: DataFrame, y: DataFrame, size: float) -> tuple:
        pass

    @abstractmethod
    def split_data_into_x_and_y(self, df: DataFrame) -> tuple:
        pass


class NormalSplitter(DataSplitter):

    # interface method implementation
    def train_test_split_data(self, x: DataFrame, y: DataFrame, size: float) -> tuple:
        # return x_train, x_test, y_train, y_test using scikitlearn train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
        temp_answer = [x_train, x_test, y_train, y_test]
        # fix outputs as pd.DataFrame if they are a pd.Series, then return them as a tuple
        mapped_answer = [i.to_frame() for i in temp_answer if isinstance(i, pd.Series)]
        return tuple(mapped_answer)

    # interface method implementation
    def split_data_into_x_and_y(self, df: DataFrame) -> tuple:
        # fix y as pd.DataFrame. By default it is a pd.Series
        y = df[df.columns[-1]].to_frame()
        x = df.drop([df.columns[-1]], axis=1)
        return x, y


class SplitterReturner:

    def __init__(self, data_splitter: DataSplitter = NormalSplitter()) -> None:
        self._data_splitter: DataSplitter = data_splitter

    def train_and_test_split(self, x: DataFrame, y: DataFrame, size: float) -> tuple:
        # if size is between valid range then return x and y train-test data
        if 0.0 < size < 1.0:
            tuple_answer = self._data_splitter.train_test_split_data(x, y, size)
            return tuple_answer
        raise ValueError("Size variable is not between 0.0 and 1.0")

    def split_x_y_from_df(self, df: DataFrame) -> tuple:
        if DataEnsurer.validate_pd_data(df):
            tuple_answer = self._data_splitter.split_data_into_x_and_y(df)
            return tuple_answer
        raise TypeError("The dataframe does no have enough samples or features to split them into x and y")
