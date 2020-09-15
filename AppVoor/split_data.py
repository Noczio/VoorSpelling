from abc import ABC
from parse_file import DataFrameParseEnsurer


class ABCDataSplitter(ABC):
    """[summary]

    Args:
        ABC ([type]): [description]
    """

    def __init__(self, df_train=None, df_test=None):
        """[summary]

        Args:
            df_train ([type], optional): [description]. Defaults to None.
            df_test ([type], optional): [description]. Defaults to None.
        """
        self._df_train = df_train
        self._df_test = df_test
        self._df_parser = DataFrameParseEnsurer()  # uses Iparse_ensurer

    def split_data(self):
        pass


class MultiDataSplitter(ABCDataSplitter):
    """[summary]

    Args:
        ABCDataSplitter ([type]): [description]
    """

    def split_data(self):
        """[summary]

        Raises:
            TypeError: [description]

        Returns:
            [type]: [description]
        """
        if (self._df_parser.is_data_correctly_parsed(self._df_train) and self._df_parser.is_data_correctly_parsed(
                self._df_test)):
            y_train = self._df_train[self._df_train.columns[-1]]
            x_train = self._df_train.drop([self._df_train.columns[-1]], axis=1)
            y_test = self._df_test[self._df_test.columns[-1]]
            x_test = self._df_test.drop([self._df_test.columns[-1]], axis=1)
            return x_train, y_train, x_test, y_test
        raise TypeError


class SimpleDataSplitter(ABCDataSplitter):
    """[summary]

    Args:
        ABCDataSplitter ([type]): [description]
    """

    def split_data(self):
        """[summary]

        Raises:
            TypeError: [description]

        Returns:
            [type]: [description]
        """
        if self._df_parser.is_data_correctly_parsed(self._df_train):
            y_train = self._df_train[self._df_train.columns[-1]]
            x_train = self._df_train.drop([self._df_train.columns[-1]], axis=1)
            return x_train, y_train
        raise TypeError
