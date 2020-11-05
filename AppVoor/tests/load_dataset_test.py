import unittest

from load_data import TSVDataLoader, DataLoaderReturner, LoaderCreator


class MyTestCase(unittest.TestCase):

    def test_data_loaded_csv(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        loader_creator = LoaderCreator.get_instance()
        csv_type = loader_creator.create_loader(test_full_path, "CSV")
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
        loader_creator = LoaderCreator.get_instance()
        tsv_type = loader_creator.create_loader(test_full_path, "TSV")
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
            loader_creator = LoaderCreator.get_instance()
            tsv_type = loader_creator.create_loader(test_full_path, "CSV")
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
            loader_creator = LoaderCreator.get_instance()
            csv_type = loader_creator.create_loader(test_full_path, "CSV")
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
            loader_creator = LoaderCreator.get_instance()
            tsv_type = loader_creator.create_loader(test_full_path, "TSV")
            # initialize data_returner with TSVDataTypeLoader
            data_returner = DataLoaderReturner(tsv_type)
            # this should raise an FileNotFoundError
            _ = data_returner.get_data()

    def test_creator_value_is_wrong(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(ValueError):
            loader_creator = LoaderCreator.get_instance()
            _ = loader_creator.create_loader(test_full_path, "txt")

    def test_creator_value_with_no_capital_letters_is_right(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        loader_creator = LoaderCreator.get_instance()
        tsv_type = loader_creator.create_loader(test_full_path, "tsv")
        # initialize data_returner with TSVDataTypeLoader
        data_returner = DataLoaderReturner(tsv_type)
        df = data_returner.get_data()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 34)

    def test_creator_value_with_no_capital_letters_and_white_space_is_right(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        loader_creator = LoaderCreator.get_instance()
        tsv_type = loader_creator.create_loader(test_full_path, " tsv ")
        # initialize data_returner with TSVDataTypeLoader
        data_returner = DataLoaderReturner(tsv_type)
        df = data_returner.get_data()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 34)

    def test_loader_creator_types_are_correct(self):
        # get LoaderCreator singleton Fabric instance
        loader_creator = LoaderCreator.get_instance()
        # check for available types
        loader_types = loader_creator.get_available_types()
        expected_types = ("CSV", "TSV", "JSON")
        bol_answer = expected_types == loader_types
        # this should assert true
        self.assertTrue(bol_answer)


if __name__ == '__main__':
    unittest.main()
