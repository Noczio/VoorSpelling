import unittest

from load_data import CSVDataTypeLoader, DataReturner
from split_data import NormalSplitter, SplitterReturner


class MyTestCase(unittest.TestCase):

    def test_single_split(self):
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = CSVDataTypeLoader(test_full_path)
        data_returner = DataReturner(csv_type)
        df = data_returner.get_data()

        expected_y_len, expected_x_len = df.shape
        # shape returns org column value, x doesn't have prediction column, so it must be org_value-1
        expected_x_len -= 1

        splitter = SplitterReturner(NormalSplitter())
        x, y = splitter.split_x_y_from_df(df)

        self.assertEqual(len(x.columns), expected_x_len)
        self.assertEqual(len(y), expected_y_len)

    def test_single_split_raise_error(self):
        with self.assertRaises(TypeError):
            df = [1, 2, 3]
            splitter = SplitterReturner(NormalSplitter())
            _, _ = splitter.split_x_y_from_df(df)


if __name__ == '__main__':
    unittest.main()
