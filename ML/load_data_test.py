import pandas as pd
import os
from load_data import CVS_data_type, TSV_data_type
import unittest

class Test_data_load(unittest.TestCase):

    def test_data_loaded_csv(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path +"\\"+folder_name+"\\"+file_name
        CSV_FILE = CVS_data_type(test_full_path)
        df = CSV_FILE.file_as_dataset()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len,9)

    def test_data_load_tsv(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = test_current_path +"\\"+folder_name+"\\"+file_name
        TSV_FILE = TSV_data_type(test_full_path)
        df = TSV_FILE.file_as_dataset()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len,34)
    
    def test_wrong_sep_for_tsv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = test_current_path +"\\"+folder_name+"\\"+file_name
        TSV_FILE = CVS_data_type(test_full_path)
        df = TSV_FILE.file_as_dataset()
        df_column_len = len(df.columns)
        self.assertNotEqual(df_column_len, 34)
    
    def test_wrong_sep_for_csv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path +"\\"+folder_name+"\\"+file_name
        CSV_FILE = TSV_data_type(test_full_path)
        df = CSV_FILE.file_as_dataset()
        df_column_len = len(df.columns)
        self.assertNotEqual(df_column_len, 9)

if __name__ == '__main__':
    unittest.main()