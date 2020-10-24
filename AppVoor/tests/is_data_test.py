import unittest

import pandas as pd

from is_data import DataEnsurer
from load_data import CSVDataTypeLoader, DataReturner, JSONDataTypeLoader


class MyTestCase(unittest.TestCase):

    def test_data_is_int(self):
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_py_data(10, int)
        self.assertTrue(ensurer_bol)

    def test_data_is_str(self):
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_py_data("test", str)
        self.assertTrue(ensurer_bol)

    def test_data_is_float(self):
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_py_data(0.25, float)
        self.assertTrue(ensurer_bol)

    def test_data_is_list(self):
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_py_data([10, "s", True], list)
        self.assertTrue(ensurer_bol)

    def test_data_is_tuple(self):
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_py_data((10, "s", True), tuple)
        self.assertTrue(ensurer_bol)

    def test_data_is_df(self):
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        csv_file = CSVDataTypeLoader(test_full_path)
        data_returner = DataReturner(csv_file)
        this_is_a_df = data_returner.get_data()
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_pd_data(this_is_a_df)
        self.assertTrue(ensurer_bol)

    def test_data_is_not_df(self):
        not_a_df = {'name': 'notch', 'job': 'dev'}
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_pd_data(not_a_df)
        self.assertFalse(ensurer_bol)

    def test_df_not_meeting_req_columns(self):
        dict_test = {'name': ['notch', 'fen', 'sky']}
        df = pd.DataFrame.from_dict(dict_test)
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_pd_data(df)
        self.assertFalse(ensurer_bol)

    def test_json_is_list(self):
        json_type = JSONDataTypeLoader(full_path=".\\..\\jsonInfo\\welcomeMessage.json")
        data_returner = DataReturner(json_type)
        file = data_returner.get_data()
        data_ensurer = DataEnsurer()
        ensurer_bol = data_ensurer.validate_py_data(file, list)
        self.assertTrue(ensurer_bol)


if __name__ == '__main__':
    unittest.main()
