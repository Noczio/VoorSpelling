import unittest

from resources.backend_scripts.estimator_creation import EstimatorCreator
from resources.backend_scripts.feature_selection import FeatureSelectorCreator
from resources.backend_scripts.load_data import LoaderCreator
from resources.backend_scripts.score import CVScore
from resources.backend_scripts.split_data import SplitterReturner


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _feature_selector_creator = FeatureSelectorCreator.get_instance()
    _estimator_creator = EstimatorCreator.get_instance()

    def test_diabetes_has_fewer_features_with_LSVC_FFS_roc_auc_10(self):
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
        # create a simple LinearSVC estimator
        clf = self._estimator_creator.create_estimator("LinearSVC")
        clf.set_params(dual=False, random_state=0)
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "roc_auc", 10)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_diabetes_has_fewer_features_with_LSVC_BFS_roc_auc_10(self):
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
        # create a simple LinearSVC estimator
        clf = self._estimator_creator.create_estimator("LinearSVC")
        clf.set_params(dual=False, random_state=0)
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "roc_auc", 10)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_diabetes_has_fewer_features_with_SVC_FFS_accuracy_10(self):
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
        # create a simple SVC estimator
        clf = self._estimator_creator.create_estimator("SVC")
        clf.set_params(random_state=0)
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "accuracy", 10)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_molecules_has_fewer_features_with_SVC_BFS_accuracy_5(self):
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
        fs = self._feature_selector_creator.create_feature_selector("BFS")
        # create a simple SVC estimator
        clf = self._estimator_creator.create_estimator("SVC")
        clf.set_params(random_state=0)
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "accuracy", 5)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_wine_quality_has_fewer_features_with_LSVR_FFS_r2_10(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "winequality-red.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        scsv_type = self._loader_creator.create_loader(test_full_path, "scsv")
        df = scsv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("FFS")
        # create a simple LSVR estimator
        clf = self._estimator_creator.create_estimator("LinearSVR")
        clf.set_params(max_iter=20000, dual=False, loss="squared_epsilon_insensitive")
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "r2", 10)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_wine_quality_has_fewer_features_with_SVR_BFS_explained_variance_5(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "winequality-white.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        scsv_type = self._loader_creator.create_loader(test_full_path, "scsv")
        df = scsv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("BFS")
        # create a simple SVR estimator
        clf = self._estimator_creator.create_estimator("SVR")
        clf.set_params(gamma="auto")
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "explained_variance", 5)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_wine_quality_has_fewer_features_with_LASSO_FFS_r2_10(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "winequality-red.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        scsv_type = self._loader_creator.create_loader(test_full_path, "scsv")
        df = scsv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("FFS")
        # create a simple Lasso estimator
        clf = self._estimator_creator.create_estimator("Lasso")
        prm = {'alpha': 1.0, 'random_state': 8, 'selection': 'cyclic', 'tol': 0.0001}
        clf.set_params(**prm)
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "r2", 10)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        print("lasso", is_fewer_than_original)

    def test_wine_quality_has_fewer_features_with_LASSO_FFS_explained_variance_10(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "winequality-red.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        scsv_type = self._loader_creator.create_loader(test_full_path, "scsv")
        df = scsv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("FFS")
        # create a simple Lasso estimator
        clf = self._estimator_creator.create_estimator("Lasso")
        prm = {'alpha': 1.0, 'random_state': 8, 'selection': 'cyclic', 'tol': 0.0001}
        clf.set_params(**prm)
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "explained_variance", 10)
        print(new_x.columns.values, f"\n{score}")
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        print("lasso", is_fewer_than_original)
        """
        ['total sulfur dioxide' 'fixed acidity' 'volatile acidity' 'citric acid'
         'residual sugar' 'chlorides' 'free sulfur dioxide' 'density' 'pH'
         'sulphates' 'alcohol'] 
        0.020356773894023884
        lasso False
        """

    def test_iris_has_fewer_features_with_KMEANS_FFS_mutual_info_score_5(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "iris.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "csv")
        df = csv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("FFS")
        # create a simple Kmeans estimator
        clf = self._estimator_creator.create_estimator("KMeans")
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "mutual_info_score", 5)
        print(new_x)
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # this should be True
        self.assertTrue(is_fewer_than_original)

    def test_iris_has_fewer_features_with_MEANSHIFT_BFS_mutual_info_score_10(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "iris.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        csv_type = self._loader_creator.create_loader(test_full_path, "csv")
        df = csv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        _, len_original_y = x.shape
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("BFS")
        # create a simple MeanShift estimator
        clf = self._estimator_creator.create_estimator("MeanShift")
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "mutual_info_score", 10)
        print(new_x.columns.values, f"\n{score}")
        _, len_new_y = new_x.shape
        # does it have fewer features?
        is_fewer_than_original: bool = True if len_new_y < len_original_y else False
        # for this dataset and estimator with bfs all of the features are necessary
        print(is_fewer_than_original)

    def test_wine_quality_with_LSVR_FFS_neg_mean_squared_error_10(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "winequality-red.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        scsv_type = self._loader_creator.create_loader(test_full_path, "scsv")
        df = scsv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("FFS")
        # create a simple LSVR estimator
        clf = self._estimator_creator.create_estimator("LinearSVR")
        clf.set_params(max_iter=20000, dual=False, loss="squared_epsilon_insensitive")
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "neg_mean_squared_error", 10)
        print(new_x.columns.values, f"\n{score}")
        """
        ['alcohol' 'volatile acidity' 'sulphates' 'chlorides'] 
        -0.4389980629892999
        """

    def test_wine_quality_with_LSVR_BFS_neg_mean_squared_error_10(self):
        # load molecules.csv from disk
        folder_name = "datasets"
        file_name = "winequality-red.csv"
        test_full_path = ".\\..\\" + folder_name + "\\" + file_name
        # get dataframe using LoaderCreator
        scsv_type = self._loader_creator.create_loader(test_full_path, "scsv")
        df = scsv_type.get_file_transformed()
        # get x and y from SplitterReturner
        x, y = SplitterReturner.split_x_y_from_df(df)
        # create a feature selector
        fs = self._feature_selector_creator.create_feature_selector("BFS")
        # create a simple LSVR estimator
        clf = self._estimator_creator.create_estimator("LinearSVR")
        clf.set_params(max_iter=20000, dual=False, loss="squared_epsilon_insensitive")
        # get new_x with new features
        new_x, score = fs.select_features(x, y, clf, "neg_mean_squared_error", 10)
        print(new_x.columns.values, f"\n{score}")
        """
        ['fixed acidity' 'volatile acidity' 'citric acid' 'residual sugar'
         'chlorides' 'free sulfur dioxide' 'total sulfur dioxide' 'density' 'pH'
         'sulphates' 'alcohol'] 
        -0.44334763280535405
        """


if __name__ == '__main__':
    unittest.main()
