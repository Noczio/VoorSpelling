from interface import Interface, implements

from sklearn.model_selection import train_test_split

from is_data import is_dataframe

import pandas as pd


class IDataSplitter(Interface):

    def train_test_split_data(self, x: pd.DataFrame, y: pd.DataFrame, size: float) -> tuple:
        pass

    def split_data_into_x_and_y(self, df: pd.DataFrame) -> tuple:
        pass


class NormalSplitter(implements(IDataSplitter)):

    def train_test_split_data(self, x: pd.DataFrame, y: pd.DataFrame, size: float) -> tuple:
        # return x_train, x_test, y_train, y_test using train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
        tuple_answer = (x_train, x_test, y_train, y_test)
        return tuple_answer

    # abstract class method implementation
    def split_data_into_x_and_y(self, df: pd.DataFrame) -> tuple:
        if is_dataframe(df):
            y = df[df.columns[-1]]
            x = df.drop([df.columns[-1]], axis=1)
            tuple_answer = (x, y)
            return tuple_answer
        raise TypeError


class SplitterReturner:

    def __init__(self, splitter: IDataSplitter) -> None:
        self._splitter = splitter

    def train_test_split(self, x: pd.DataFrame, y: pd.DataFrame, size: float) -> tuple:
        tuple_answer = self._splitter.train_test_split_data(x, y, size)
        return tuple_answer

    def split_x_y_from_df(self, df: pd.DataFrame) -> tuple:
        tuple_answer = self._splitter.split_data_into_x_and_y(df)
        return tuple_answer
