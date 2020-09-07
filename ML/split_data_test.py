import pandas as pd
import os
from load_dataset import CVS_data_type_loader, TSV_data_type_loader
from split_data import Simple_data_splitter, Multi_data_splitter
import unittest

class Test_split_data(unittest.TestCase):
    def test_single_split(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path +"\\"+folder_name+"\\"+file_name
        CSV_FILE = CVS_data_type_loader(test_full_path)
        df = CSV_FILE.file_as_dataset()

        expected_y_len, expected_x_len = df.shape
        expected_x_len-=1 # shape returns org column value, x doesn't have prediction column, so it must be org_value-1

        simple_splitter = Simple_data_splitter(df_train=df)
        x,y = simple_splitter.split_data()

        self.assertEqual(len(x.columns),expected_x_len)
        self.assertEqual(len(y),expected_y_len)   

    def test_multi_split(self):
        test_current_path = os.getcwd()
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = test_current_path +"\\"+folder_name+"\\"+file_name
        CSV_FILE = CVS_data_type_loader(test_full_path)
        df = CSV_FILE.file_as_dataset()

        training_part_df = df.sample(frac = 0.8) 
        test_part_df = df.drop(training_part_df.index) 

        expected_y_train_len, expected_x_train_len = training_part_df.shape
        expected_x_train_len-=1 # shape returns org column value, x doesn't have prediction column, so it must be org_value-1

        expected_y_test_len, expected_x_test_len = test_part_df.shape
        expected_x_test_len-=1 # shape returns org column value, x doesn't have prediction column, so it must be org_value-1

        multi_splitter = Multi_data_splitter(df_train=training_part_df,df_test=test_part_df)
        x_train,y_train,x_test,y_test = multi_splitter.split_data()

        self.assertEqual(len(x_train.columns),expected_x_train_len)
        self.assertEqual(len(list(y_train)),expected_y_train_len)
        self.assertEqual(len(x_test.columns),expected_x_test_len)
        self.assertEqual(len(list(y_test)),expected_y_test_len)
 
    def test_multi_split_raise_error(self):
        with self.assertRaises(TypeError):
            df_train = [1,2,3]
            df_test = "not expected type"
            multi_splitter = Multi_data_splitter(df_train=df_train,df_test=df_test)
            x_train,y_train,x_test,y_test = multi_splitter.split_data()

    def test_single_split_raise_error(self):
        with self.assertRaises(TypeError):
            df = [1,2,3]
            simple_splitter = Simple_data_splitter(df_train=df)
            x,y = simple_splitter.split_data()

if __name__ == '__main__':
    unittest.main()