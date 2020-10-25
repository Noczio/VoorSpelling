import unittest

from is_data import DataEnsurer
from load_data import CSVDataTypeLoader, DataReturner
from split_data import NormalSplitter, SplitterReturner
import pandas as pd


class MyTestCase(unittest.TestCase):

    def test_single_split_columns_match(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = CSVDataTypeLoader(test_full_path)
        # initialize data_returner with CSVDataTypeLoader
        data_returner = DataReturner(csv_type)
        df = data_returner.get_data()
        expected_y_len, expected_x_len = df.shape  # true prediction and data len with shape method
        # shape returns original column value. x doesn't have prediction column, so it must be original value - 1
        expected_x_len -= 1
        # use of splitterReturner with a NormalSplitter implementation
        splitter = SplitterReturner(NormalSplitter())
        x, y = splitter.split_x_y_from_df(df)
        # do the values match in both x and y dataframes
        self.assertEqual(len(x.columns), expected_x_len)
        self.assertEqual(len(y), expected_y_len)

    def test_single_split_returns_a_tuple(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = CSVDataTypeLoader(test_full_path)
        # initialize data_returner with CSVDataTypeLoader
        data_returner = DataReturner(csv_type)
        df = data_returner.get_data()
        # use of splitterReturner with a NormalSplitter implementation
        splitter = SplitterReturner(NormalSplitter())
        # split dataframe into x and y
        data = splitter.split_x_y_from_df(df)
        result = DataEnsurer.validate_py_data(data, tuple)
        self.assertTrue(result)

    def test_single_split_x_and_y_are_a_dataframe(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = CSVDataTypeLoader(test_full_path)
        # initialize data_returner with CSVDataTypeLoader
        data_returner = DataReturner(csv_type)
        df = data_returner.get_data()
        # use of splitterReturner with a NormalSplitter implementation
        splitter = SplitterReturner(NormalSplitter())
        # split dataframe into x and y
        data = splitter.split_x_y_from_df(df)
        results = [isinstance(d, pd.DataFrame) for d in data]
        # are all outputs True?
        for r in results:
            self.assertTrue(r)

    def test_train_test_split_data_all_data_is_dataframe(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = CSVDataTypeLoader(test_full_path)
        # initialize data_returner with CSVDataTypeLoader
        data_returner = DataReturner(csv_type)
        df = data_returner.get_data()
        # use of splitterReturner with a NormalSplitter implementation
        splitter = SplitterReturner(NormalSplitter())
        # split dataframe into x and y, then use train_and_test_split
        x, y = splitter.split_x_y_from_df(df)
        data = splitter.train_and_test_split(x, y, 0.2)  # 80 percent of data should be training and the other 20 is
        # test data
        # map all data from 0 to 2 (x_train, x_test) and check if it is a dataframe
        results = [isinstance(d, pd.DataFrame) for d in data]
        # are all outputs True?
        for r in results:
            self.assertTrue(r)


if __name__ == '__main__':
    unittest.main()
