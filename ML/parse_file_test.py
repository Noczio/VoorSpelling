import pandas as pd
import os
import unittest
from parse_file import DataFrameParseEnsurer
from load_dataset import CVSDataTypeLoader


class Test_split_data(unittest.TestCase):
    def test_data_is_df(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path + "\\"+folder_name+"\\"+file_name
        CSV_FILE = CVSDataTypeLoader(test_full_path)
        this_is_a_df = CSV_FILE.get_file_as_dataframe()
        df_parser = DataFrameParseEnsurer()
        parser_answ = df_parser.is_data_correctly_parsed(this_is_a_df)
        self.assertTrue(parser_answ)

    def test_data_is_not_df(self):
        not_a_df = {'name': 'nocz', 'job': 'dev'}
        df_parser = DataFrameParseEnsurer()
        parser_answ = df_parser.is_data_correctly_parsed(not_a_df)
        self.assertFalse(parser_answ)

    def test_df_not_meeting_req_columns(self):
        dict_test = {'name': ['nocz', 'fene', 'sky']}
        df = pd.DataFrame.from_dict(dict_test)
        df_parser = DataFrameParseEnsurer()
        parser_answ = df_parser.is_data_correctly_parsed(df)
        self.assertFalse(parser_answ)


if __name__ == '__main__':
    unittest.main()
