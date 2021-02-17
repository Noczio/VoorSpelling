import unittest

from backend_scripts.load_data import LoaderCreator


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

    def test_data_loaded_scsv(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "winequality-red.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        scsv_type = self._loader_creator.create_loader(test_full_path, "SCSV")
        df = scsv_type.get_file_transformed()
        df_column_len = len(df.columns)
        # do the values match?
        self.assertEqual(df_column_len, 12)

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

    def test_wrong_sep_for_scsv_file(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "winequality-red.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(TypeError):
            # get dataframe using LoaderCreator
            scsv_type = self._loader_creator.create_loader(test_full_path, "CSV")
            # this should raise an TypeError
            _ = scsv_type.get_file_transformed()

    def test_wrong_sep_for_csv_file(self):
        # load diabetes.csv from disk
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

    def test_wrong_path_scsv_file(self):
        # load mol.csv from disk. This file does not exist
        folder_name = "datasets"
        file_name = "mol.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(FileNotFoundError):
            # get dataframe using LoaderCreator
            scsv_type = self._loader_creator.create_loader(test_full_path, "SCSV")
            # this should raise an FileNotFoundError
            _ = scsv_type.get_file_transformed()

    def test_creator_value_is_wrong(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(AttributeError):
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
        expected_types = ("CSV", "TSV", "JSON", "SCSV")
        results = [True for i in loader_types if i in expected_types]
        bol_answer = all(results)
        # this should assert true
        self.assertTrue(bol_answer)

    def test_file_is_not_a_dataset_it_is_a_json(self):
        # load mol.csv from disk. This file does not exist
        folder_name = "jsonInfo"
        file_name = "helpMessage.json"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        with self.assertRaises(Exception):
            # get dataframe using LoaderCreator
            csv_type = self._loader_creator.create_loader(test_full_path, "CSV")
            # this should raise an FileNotFoundError
            _ = csv_type.get_file_transformed()


if __name__ == '__main__':
    unittest.main()
