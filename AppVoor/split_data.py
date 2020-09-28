from abc import ABC, abstractmethod
from parse_file import DataParseEnsurer
from sklearn.model_selection import train_test_split


class ABCDataSplitter(ABC):
    def __init__(self, df=None):
        self._df = df

    @abstractmethod
    def split_data_into_x_and_y(self):
        pass

    def train_test_split_data(self, x, y, size: float):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size)
        return x_train, x_test, y_train, y_test


class DataSplitter(ABCDataSplitter):
    def split_data_into_x_and_y(self):
        if DataParseEnsurer.is_dataframe(self._df):
            y = self._df[self._df.columns[-1]]
            x = self._df.drop([self._df.columns[-1]], axis=1)
            return x, y
        raise TypeError
