import os

from supervised import AutoML

from load_dataset import CSVDataTypeLoader
from sklearn.model_selection import train_test_split
from split_data import SimpleDataSplitter

test_current_path = os.getcwd()
folder_name = "datasets"
file_name = "diabetes.csv"
test_full_path = test_current_path + "\\" + folder_name + "\\" + file_name
csv_file = CSVDataTypeLoader(test_full_path)
this_is_a_df = csv_file.get_file_as_dataframe()
data_splitter = SimpleDataSplitter(df_train=this_is_a_df)
x, y = data_splitter.split_data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

automl = AutoML(
    explain_level=0,
    validation_strategy={
        "validation_type": "kfold",
        "k_folds": 10,
        "shuffle": False,
        "stratify": True,
    })
# automl.fit(x_train, y_train)
# predictions = automl.predict(x_test)
# print(predictions)
