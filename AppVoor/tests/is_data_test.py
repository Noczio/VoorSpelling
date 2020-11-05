import unittest

import pandas as pd

from is_data import DataEnsurer
from load_data import CSVDataLoader, DataLoaderReturner, JSONDataLoader


class MyTestCase(unittest.TestCase):

    def test_data_is_int(self):
        # is 10 an int?
        ensurer_bol = DataEnsurer.validate_py_data(10, int)
        self.assertTrue(ensurer_bol)

    def test_data_is_str(self):
        # is "test" a string?
        ensurer_bol = DataEnsurer.validate_py_data("test", str)
        self.assertTrue(ensurer_bol)

    def test_data_is_float(self):
        # is 0.25 a float?
        ensurer_bol = DataEnsurer.validate_py_data(0.25, float)
        self.assertTrue(ensurer_bol)

    def test_data_is_list(self):
        # is [10, "s", True] a list?
        ensurer_bol = DataEnsurer.validate_py_data([10, "s", True], list)
        self.assertTrue(ensurer_bol)

    def test_data_is_tuple(self):
        # is (10, "s", True) a list?
        ensurer_bol = DataEnsurer.validate_py_data((10, "s", True), tuple)
        self.assertTrue(ensurer_bol)

    def test_data_is_df(self):
        # load diabetes.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_file = CSVDataLoader(test_full_path)
        # initialize data_returner with CSVDataTypeLoader
        data_returner = DataLoaderReturner(csv_file)
        # get the dataframe from the data_returner
        this_is_a_df = data_returner.get_data()
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
        json_type = JSONDataLoader(file_path=".\\..\\jsonInfo\\welcomeMessage.json")
        # initialize data_returner with JSONDataTypeLoader
        data_returner = DataLoaderReturner(json_type)
        file = data_returner.get_data()
        # is the file a deserialized json list?
        ensurer_bol = DataEnsurer.validate_py_data(file, list)
        # this should be true, since welcomeMessage.json has list format
        self.assertTrue(ensurer_bol)


if __name__ == '__main__':
    unittest.main()
