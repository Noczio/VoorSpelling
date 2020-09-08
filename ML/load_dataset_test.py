import pandas as pd
import os
from load_dataset import CVSDataTypeLoader, TSVDataTypeLoader
import unittest


class Test_data_load(unittest.TestCase):

    def test_data_loaded_csv(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path + "\\"+folder_name+"\\"+file_name
        CSV_FILE = CVSDataTypeLoader(test_full_path)
        df = CSV_FILE.get_file_as_dataframe()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len, 9)

    def test_data_load_tsv(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = test_current_path + "\\"+folder_name+"\\"+file_name
        TSV_FILE = TSVDataTypeLoader(test_full_path)
        df = TSV_FILE.get_file_as_dataframe()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len, 34)

    def test_wrong_sep_for_tsv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = test_current_path + "\\"+folder_name+"\\"+file_name
        with self.assertRaises(TypeError):
            TSV_FILE = CVSDataTypeLoader(test_full_path)
            df = TSV_FILE.get_file_as_dataframe()

    def test_wrong_sep_for_csv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path + "\\"+folder_name+"\\"+file_name
        with self.assertRaises(TypeError):
            CSV_FILE = TSVDataTypeLoader(test_full_path)
            df = CSV_FILE.get_file_as_dataframe()
            df_column_len = len(df.columns)

    def test_wrong_path_csv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = test_current_path + "\\"+folder_name+"\\"+file_name
        with self.assertRaises(FileNotFoundError):
            CSV_FILE = CVSDataTypeLoader(test_full_path)
            df = CSV_FILE.get_file_as_dataframe()

    def test_wrong_path_tsv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = test_current_path + "\\"+folder_name+"\\"+file_name
        with self.assertRaises(FileNotFoundError):
            TSV_FILE = TSVDataTypeLoader(test_full_path)
            df = TSV_FILE.get_file_as_dataframe()


if __name__ == '__main__':
    unittest.main()
