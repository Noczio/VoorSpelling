from abc import abstractmethod, ABC

import pandas as pd
from sklearn.model_selection import train_test_split

from is_data import DataEnsurer

DataFrame = pd.DataFrame


class IDataSplitter(ABC):

    @abstractmethod
    def train_test_split_data(self, x: DataFrame, y: DataFrame, size: float) -> tuple:
        pass

    @abstractmethod
    def split_data_into_x_and_y(self, df: DataFrame) -> tuple:
        pass


class NormalSplitter(IDataSplitter):

    # interface method implementation
    def train_test_split_data(self, x: DataFrame, y: DataFrame, size: float) -> tuple:
        # return x_train, x_test, y_train, y_test using train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
        tuple_answer = (x_train, x_test, y_train, y_test)
        return tuple_answer

    # interface method implementation
    def split_data_into_x_and_y(self, df: DataFrame) -> tuple:
        data_ensurer = DataEnsurer()
        if data_ensurer.validate_pd_data(df):
            y = df[df.columns[-1]]
            x = df.drop([df.columns[-1]], axis=1)
            tuple_answer = (x, y)
            return tuple_answer
        raise TypeError


class SplitterReturner:

    def __init__(self, data_splitter: IDataSplitter) -> None:
        self._data_splitter = data_splitter

    def train_and_test_split(self, x: DataFrame, y: DataFrame, size: float) -> tuple:
        tuple_answer = self._data_splitter.train_test_split_data(x, y, size)
        return tuple_answer

    def split_x_y_from_df(self, df: DataFrame) -> tuple:
        tuple_answer = self._data_splitter.split_data_into_x_and_y(df)
        return tuple_answer
