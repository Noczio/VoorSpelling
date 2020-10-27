import unittest

from load_data import CSVDataLoader, TSVDataLoader, DataLoaderReturner


class MyTestCase(unittest.TestCase):

    def test_data_loaded_csv(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_type = CSVDataLoader(test_full_path)
        # initialize data_returner with CSVDataTypeLoader
        data_returner = DataLoaderReturner(csv_type)
        df = data_returner.get_data()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 9)

    def test_data_load_tsv(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        tsv_type = TSVDataLoader(test_full_path)
        # initialize data_returner with TSVDataTypeLoader
        data_returner = DataLoaderReturner(tsv_type)
        df = data_returner.get_data()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 34)

    def test_wrong_sep_for_tsv_file(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            tsv_type = CSVDataLoader(test_full_path)
            # initialize data_returner with CSVDataTypeLoader
            data_returner = DataLoaderReturner(tsv_type)
            # this should raise an TypeError
            _ = data_returner.get_data()

    def test_wrong_sep_for_csv_file(self):
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            csv_type = TSVDataLoader(test_full_path)
            # initialize data_returner with TSVDataTypeLoader
            data_returner = DataLoaderReturner(csv_type)
            # this should raise an TypeError
            _ = data_returner.get_data()

    def test_wrong_path_csv_file(self):
        # load mol.csv from disk. This file does not exist
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            csv_type = CSVDataLoader(test_full_path)
            # initialize data_returner with CSVDataTypeLoader
            data_returner = DataLoaderReturner(csv_type)
            # this should raise an FileNotFoundError
            _ = data_returner.get_data()

    def test_wrong_path_tsv_file(self):
        # load mol.csv from disk. This file does not exist
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            tsv_type = TSVDataLoader(test_full_path)
            # initialize data_returner with TSVDataTypeLoader
            data_returner = DataLoaderReturner(tsv_type)
            # this should raise an FileNotFoundError
            _ = data_returner.get_data()


if __name__ == '__main__':
    unittest.main()
