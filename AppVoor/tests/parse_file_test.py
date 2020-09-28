import os
import unittest

import pandas as pd

from load_dataset import CSVDataTypeLoader
from parse_file import DataParseEnsurer


class MyTestCase(unittest.TestCase):

    def test_data_is_df(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        csv_file = CSVDataTypeLoader(test_full_path)
        this_is_a_df = csv_file.get_file_as_dataframe()
        parser_answer = DataParseEnsurer.is_dataframe(this_is_a_df)
        self.assertTrue(parser_answer)

    def test_data_is_not_df(self):
        not_a_df = {'name': 'notch', 'job': 'dev'}
        parser_answer = DataParseEnsurer.is_dataframe(not_a_df)
        self.assertFalse(parser_answer)

    def test_df_not_meeting_req_columns(self):
        dict_test = {'name': ['notch', 'fen', 'sky']}
        df = pd.DataFrame.from_dict(dict_test)
        parser_answer = DataParseEnsurer.is_dataframe(df)
        self.assertFalse(parser_answer)


if __name__ == '__main__':
    unittest.main()
