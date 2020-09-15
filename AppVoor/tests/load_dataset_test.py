import os
import unittest

from load_dataset import CSVDataTypeLoader, TSVDataTypeLoader


class MyTestCase(unittest.TestCase):
    def test_data_loaded_csv(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        csv_file = CSVDataTypeLoader(test_full_path)
        df = csv_file.get_file_as_dataframe()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len, 9)

    def test_data_load_tsv(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        tsv_file = TSVDataTypeLoader(test_full_path)
        df = tsv_file.get_file_as_dataframe()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len, 34)

    def test_wrong_sep_for_tsv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            tsv_file = CSVDataTypeLoader(test_full_path)
            _ = tsv_file.get_file_as_dataframe()

    def test_wrong_sep_for_csv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            csv_file = TSVDataTypeLoader(test_full_path)
            df = csv_file.get_file_as_dataframe()
            _ = len(df.columns)

    def test_wrong_path_csv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            csv_file = CSVDataTypeLoader(test_full_path)
            _ = csv_file.get_file_as_dataframe()

    def test_wrong_path_tsv_file(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = test_current_path + "\\.." + "\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            tsv_file = TSVDataTypeLoader(test_full_path)
            _ = tsv_file.get_file_as_dataframe()


if __name__ == '__main__':
    unittest.main()
