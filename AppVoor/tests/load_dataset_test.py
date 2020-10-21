import unittest

from load_data import CSVDataTypeLoader, TSVDataTypeLoader, DataReturner


class MyTestCase(unittest.TestCase):

    def test_data_loaded_csv(self):
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = CSVDataTypeLoader(test_full_path)
        data_returner = DataReturner(csv_type)

        df = data_returner.get_dataframe()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len, 9)

    def test_data_load_tsv(self):
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        tsv_type = TSVDataTypeLoader(test_full_path)
        data_returner = DataReturner(tsv_type)
        df = data_returner.get_dataframe()
        df_column_len = len(df.columns)
        self.assertEqual(df_column_len, 34)

    def test_wrong_sep_for_tsv_file(self):
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            tsv_type = CSVDataTypeLoader(test_full_path)
            data_returner = DataReturner(tsv_type)
            _ = data_returner.get_dataframe()

    def test_wrong_sep_for_csv_file(self):
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            csv_type = TSVDataTypeLoader(test_full_path)
            data_returner = DataReturner(csv_type)
            _ = data_returner.get_dataframe()

    def test_wrong_path_csv_file(self):
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            csv_type = CSVDataTypeLoader(test_full_path)
            data_returner = DataReturner(csv_type)
            _ = data_returner.get_dataframe()

    def test_wrong_path_tsv_file(self):
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            tsv_type = TSVDataTypeLoader(test_full_path)
            data_returner = DataReturner(tsv_type)
            _ = data_returner.get_dataframe()


if __name__ == '__main__':
    unittest.main()
