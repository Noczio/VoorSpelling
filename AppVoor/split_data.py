from abc import ABC, abstractmethod

from sklearn.model_selection import train_test_split

from is_data import is_dataframe


class ABCDataSplitter(ABC):

    def __init__(self, df=None):
        # by default _df is None
        self._df = df

    def train_test_split_data(self, x, y, size: float):
        # return x_train, x_test, y_train, y_test using train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
        return x_train, x_test, y_train, y_test

    @abstractmethod
    def split_data_into_x_and_y(self):
        pass


class DataSplitter(ABCDataSplitter):

    # abstract class method implementation
    def split_data_into_x_and_y(self):
        if is_dataframe(self._df):
            y = self._df[self._df.columns[-1]]
            x = self._df.drop([self._df.columns[-1]], axis=1)
            return x, y
        raise TypeError
