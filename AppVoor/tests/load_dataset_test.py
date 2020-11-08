import unittest

from load_data import LoaderCreator


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()

    def test_data_loaded_csv(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
        df = csv_type.get_file_transformed()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 9)

    def test_data_load_tsv(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        tsv_type = self._loader_creator.create_loader(test_full_path, "TSV")
        df = tsv_type.get_file_transformed()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 34)

    def test_wrong_sep_for_tsv_file(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            # get dataframe using LoaderCreator
            tsv_type = self._loader_creator.create_loader(test_full_path, "CSV")
            # this should raise an TypeError
            _ = tsv_type.get_file_transformed()

    def test_wrong_sep_for_csv_file(self):
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            loader_creator = LoaderCreator.get_instance()
            csv_type = loader_creator.create_loader(test_full_path, "TSV")
            # this should raise an TypeError
            _ = csv_type.get_file_transformed()

    def test_wrong_path_csv_file(self):
        # load mol.csv from disk. This file does not exist
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            # get dataframe using LoaderCreator
            csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
            # this should raise an FileNotFoundError
            _ = csv_type.get_file_transformed()

    def test_wrong_path_tsv_file(self):
        # load mol.csv from disk. This file does not exist
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            # get dataframe using LoaderCreator
            tsv_type = self._loader_creator.create_loader(test_full_path, "TSV")
            # this should raise an FileNotFoundError
            _ = tsv_type.get_file_transformed()

    def test_creator_value_is_wrong(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(ValueError):
            # get dataframe using LoaderCreator. This should raise a ValueError
            _ = self._loader_creator.create_loader(test_full_path, "txt")

    def test_creator_value_with_no_capital_letters_is_right(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        tsv_type = self._loader_creator.create_loader(test_full_path, "tsv")
        df = tsv_type.get_file_transformed()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 34)

    def test_creator_value_with_no_capital_letters_and_white_space_is_right(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        tsv_type = self._loader_creator.create_loader(test_full_path, " tsv ")
        df = tsv_type.get_file_transformed()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 34)

    def test_loader_creator_types_are_correct(self):
        # check for available types
        loader_types = self._loader_creator.get_available_types()
        expected_types = ("CSV", "TSV", "JSON")
        bol_answer = expected_types == loader_types
        # this should assert true
        self.assertTrue(bol_answer)


if __name__ == '__main__':
    unittest.main()
