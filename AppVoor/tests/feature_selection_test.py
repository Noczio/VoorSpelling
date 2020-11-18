import unittest

from feature_selection import FeatureSelectorCreator
from load_data import LoaderCreator
from split_data import SplitterReturner

from sklearn.svm import LinearSVC, SVC


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _feature_selector_creator = FeatureSelectorCreator.get_instance()

    def test_molecules_has_fewer_features_with_LSVC_SFM(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "molecules.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        tsv_type = self._loader_creator.create_loader(test_full_path, "tsv")
        df = tsv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        x = x.drop(["m_name"], axis=1)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("SFM")
        # create a simple LinearSVC
        clf = LinearSVC(C=3, tol=0.0001, random_state=0, dual=False)
        # get new_x with new features
        new_x = fs.select_features(x, y, clf)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_diabetes_has_fewer_features_with_LSVC_RFE(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "csv")
        df = csv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("RFE")
        # create a simple LinearSVC
        clf = LinearSVC(random_state=0, dual=False)
        # get new_x with new features
        new_x = fs.select_features(x, y, clf)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_diabetes_has_fewer_features_with_LSVC_FFS(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "csv")
        df = csv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("FFS")
        # create a simple LinearSVC
        clf = LinearSVC(random_state=0, dual=False)
        # get new_x with new features
        new_x = fs.select_features(x, y, clf)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_diabetes_has_fewer_features_with_LSVC_BFS(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "csv")
        df = csv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("BFS")
        # create a simple LinearSVC
        clf = LinearSVC(random_state=0, dual=False)
        # get new_x with new features
        new_x = fs.select_features(x, y, clf)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_diabetes_has_fewer_features_with_SVC_FFS(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "csv")
        df = csv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("FFS")
        # create a simple LinearSVC
        clf = SVC(random_state=0)
        # get new_x with new features
        new_x = fs.select_features(x, y, clf)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_diabetes_has_fewer_features_with_SVC_BFS(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "diabetes.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "csv")
        df = csv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("BFS")
        # create a simple LinearSVC
        clf = SVC(random_state=0)
        # get new_x with new features
        new_x = fs.select_features(x, y, clf)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)


if __name__ == '__main__':
    unittest.main()
