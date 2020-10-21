import unittest

import pandas as pd

from is_data import IsData
from load_data import CSVDataTypeLoader, DataReturner


class MyTestCase(unittest.TestCase):

    def test_data_is_int(self):
        ensurer_bol = IsData.is_int(10)
        self.assertTrue(ensurer_bol)

    def test_data_is_str(self):
        ensurer_bol = IsData.is_str("test")
        self.assertTrue(ensurer_bol)

    def test_data_is_float(self):
        ensurer_bol = IsData.is_float(0.25)
        self.assertTrue(ensurer_bol)

    def test_data_is_list(self):
        ensurer_bol = IsData.is_list([10, "s", True])
        self.assertTrue(ensurer_bol)

    def test_data_is_tuple(self):
        ensurer_bol = IsData.is_tuple((10, "s", True))
        self.assertTrue(ensurer_bol)

    def test_data_is_df(self):
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_file = CSVDataTypeLoader(test_full_path)
        data_returner = DataReturner(csv_file)
        this_is_a_df = data_returner.get_dataframe()
        parser_answer = IsData.is_dataframe(this_is_a_df)
        self.assertTrue(parser_answer)

    def test_data_is_not_df(self):
        not_a_df = {'name': 'notch', 'job': 'dev'}
        parser_answer = IsData.is_dataframe(not_a_df)
        self.assertFalse(parser_answer)

    def test_df_not_meeting_req_columns(self):
        dict_test = {'name': ['notch', 'fen', 'sky']}
        df = pd.DataFrame.from_dict(dict_test)
        parser_answer = IsData.is_dataframe(df)
        self.assertFalse(parser_answer)


if __name__ == '__main__':
    unittest.main()
