import unittest

import pandas as pd
import numpy as np

from resources.backend_scripts.is_data import DataEnsurer
from resources.backend_scripts.load_data import LoaderCreator
from resources.backend_scripts.split_data import SplitterReturner


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator()

    def test_single_split_columns_match(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
        df = csv_type.get_file_transformed()
        expected_y_len, expected_x_len = df.shape  # true prediction and data len with shape method
        # shape returns original column value. x doesn't have prediction column, so it must be original value - 1
        expected_x_len -= 1
        # use of splitterReturner with a NormalSplitter implementation
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # do the values match in both x and y dataframes
        self.assertEqual(len(x.columns), expected_x_len)
        self.assertEqual(len(y), expected_y_len)

    def test_single_split_returns_a_tuple(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
        df = csv_type.get_file_transformed()
        # use of splitterReturner with a NormalSplitter implementation
        splitter = SplitterReturner()
        # split dataframe into x and y
        data = splitter.split_x_y_from_df(df)
        result = DataEnsurer.validate_py_data(data, tuple)
        self.assertTrue(result)

    def test_single_split_x_and_y_is_a_dataframe_and_numpy_array(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
        df = csv_type.get_file_transformed()
        # use of splitterReturner with a NormalSplitter implementation
        splitter = SplitterReturner()
        # split dataframe into x and y
        data = splitter.split_x_y_from_df(df)
        results = [isinstance(data[0], pd.DataFrame), isinstance(data[-1], np.ndarray)]
        # are all outputs True?
        for r in results:
            self.assertTrue(r)

    def test_train_test_split_size_zero_is_wrong(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
        df = csv_type.get_file_transformed()
        # use of splitterReturner with a NormalSplitter implementation
        with self.assertRaises(ValueError):
            splitter = SplitterReturner()
            # split dataframe into x and y, then use train_and_test_split
            x, y = splitter.split_x_y_from_df(df)
            _ = splitter.train_and_test_split(x, y, 0.0)  # 80 percent of data should be training and the other 20 is

    def test_train_test_split_size_less_than_zero_is_wrong(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
        df = csv_type.get_file_transformed()
        # this should raise a ValueError because size = -0.5 is not a valid number
        with self.assertRaises(ValueError):
            # use of splitterReturner with a NormalSplitter implementation
            splitter = SplitterReturner()
            # split dataframe into x and y, then use train_and_test_split
            x, y = splitter.split_x_y_from_df(df)
            _ = splitter.train_and_test_split(x, y, -0.5)  # -0.5 is not a valid value

    def test_split_into_x_and_y_is_not_a_valid_dataframe(self):
        # dummy dictionary
        temp_dict = {'x': [i for i in range(200)]}
        # transform dictionary to dataframe
        df = pd.DataFrame.from_dict(temp_dict)
        # this should raise a TypeError because dataframe doesnt meet column requirements
        with self.assertRaises(TypeError):
            splitter = SplitterReturner()
            _, _ = splitter.split_x_y_from_df(df)


if __name__ == '__main__':
    unittest.main()
