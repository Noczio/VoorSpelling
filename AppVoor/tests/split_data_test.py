import os
import unittest

from load_data import CSVDataTypeLoader
from split_data import DataSplitter


class MyTestCase(unittest.TestCase):

    def test_single_split(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        csv_file = CSVDataTypeLoader(test_full_path)
        df = csv_file.get_file_transformed()

        expected_y_len, expected_x_len = df.shape
        # shape returns org column value, x doesn't have prediction column, so it must be org_value-1
        expected_x_len -= 1

        splitter = DataSplitter(df)
        x, y = splitter.split_data_into_x_and_y()

        self.assertEqual(len(x.columns), expected_x_len)
        self.assertEqual(len(y), expected_y_len)

    def test_single_split_raise_error(self):
        with self.assertRaises(TypeError):
            df = [1, 2, 3]
            splitter = DataSplitter(df)
            _, _ = splitter.split_data_into_x_and_y()


if __name__ == '__main__':
    unittest.main()
