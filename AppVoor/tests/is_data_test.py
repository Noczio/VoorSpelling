import unittest

import pandas as pd

from backend_scripts.is_data import DataEnsurer
from jsonInfo.welcome import WelcomeMessage
from backend_scripts.load_data import LoaderCreator


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()

    def test_data_is_df(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_file = self._loader_creator.create_loader(test_full_path, "csv")
        # get the dataframe from the data_returner
        this_is_a_df = csv_file.get_file_transformed()
        # use DataEnsurer and check if it is a dataframe with enough samples and features
        ensurer_bol = DataEnsurer.validate_pd_data(this_is_a_df)
        self.assertTrue(ensurer_bol)

    def test_data_is_not_df(self):
        not_a_df = {'name': 'notch', 'job': 'dev'}
        # is {'name': 'notch', 'job': 'dev'} a dataframe?
        ensurer_bol = DataEnsurer.validate_pd_data(not_a_df)
        # it should be false, since input is a dict
        self.assertFalse(ensurer_bol)

    def test_df_not_meeting_req_columns(self):
        dict_test = {'name': [str(i) + "name" for i in range(200)]}
        df = pd.DataFrame.from_dict(dict_test)
        # is {'name': ['0name', '1name', '2name' ...]} a dataframe after pd.DataFrame.from_dict ?
        ensurer_bol = DataEnsurer.validate_pd_data(df)
        # it should be false, since it doesnt have enough samples and features
        self.assertFalse(ensurer_bol)

    def test_json_is_list(self):
        json_type = self._loader_creator.create_loader(".\\..\\jsonInfo\\welcomeMessage.json", "json")
        file = json_type.get_file_transformed()
        # is the file a deserialized json list?
        ensurer_bol = DataEnsurer.validate_py_data(file, list)
        # this should be true, since welcomeMessage.json has list format
        self.assertTrue(ensurer_bol)

    def test_loader_path_is_str(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_file = self._loader_creator.create_loader(test_full_path, "csv")
        path = csv_file.file_path
        bol_answer = DataEnsurer.validate_py_data(path, str)
        self.assertTrue(bol_answer)

    def test_data_type_is_list(self):
        # initialization of welcome_message with its path
        welcome_message = WelcomeMessage(file_path=".\\..\\jsonInfo\\welcomeMessage.json", data_type=list)
        data = welcome_message.data  # get data value using its property
        bol_answer = DataEnsurer.validate_py_data(data, list)
        # is data a list?
        self.assertTrue(bol_answer)

    def test_data_is_a_corrupted_file_csv(self):
        with self.assertRaises(TypeError):
            # load diabetes.csv from disk
            folder_name = "datasets"
            file_name = "corrupted_file_test.txt"
            test_full_path = ".\\..\\" + folder_name + "\\" + file_name
            csv_file = self._loader_creator.create_loader(test_full_path, "csv")
            # get the dataframe from the data_returner
            this_is_not_a_df = csv_file.get_file_transformed()

    def test_data_is_a_corrupted_file_tsv(self):
        with self.assertRaises(TypeError):
            # load diabetes.csv from disk
            folder_name = "datasets"
            file_name = "corrupted_file_test.txt"
            test_full_path = ".\\..\\" + folder_name + "\\" + file_name
            csv_file = self._loader_creator.create_loader(test_full_path, "tsv")
            # get the dataframe from the data_returner
            this_is_not_a_df = csv_file.get_file_transformed()


if __name__ == '__main__':
    unittest.main()
