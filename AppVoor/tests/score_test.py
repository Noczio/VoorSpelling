import unittest

from resources.backend_scripts.estimator_creation import EstimatorCreator
from resources.backend_scripts.load_data import LoaderCreator
from resources.backend_scripts.score import CVScore
from resources.backend_scripts.split_data import SplitterReturner


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator()
    _estimator_creator = EstimatorCreator()

    def test_score_type_raises_ValueError(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple linearSVC estimator
        model = self._estimator_creator.create_estimator("LinearSVC")
        model.set_params(dual=False, random_state=0)
        with self.assertRaises(ValueError):
            # get score from a linearSVC estimator with roc_auc score and 10 folds
            _ = cv_score.get_score(x, y, model, "roc", 10)

    def test_n_folds_validation_raises_ValueError(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple linearSVC estimator
        model = self._estimator_creator.create_estimator("LinearSVC")
        model.set_params(dual=False, random_state=0)
        with self.assertRaises(ValueError):
            # get score from a linearSVC estimator with roc_auc score and 10 folds
            _ = cv_score.get_score(x, y, model, "roc_auc", 2)

    def test_n_folds_validation_and_score_type_raises_ValueError(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple linearSVC estimator
        model = self._estimator_creator.create_estimator("LinearSVC")
        model.set_params(dual=False, random_state=0)
        with self.assertRaises(ValueError):
            # get score from a linearSVC estimator with roc_auc score and 10 folds
            _ = cv_score.get_score(x, y, model, "roc", 2)

    def test_cv_score_is_more_than_zero_with_LSVC_SVC_KNN_GNB_accuracy_5(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple a svc, knn and gnb estimator
        model_1 = self._estimator_creator.create_estimator("SVC")
        model_2 = self._estimator_creator.create_estimator("KNeighborsClassifier")
        model_3 = self._estimator_creator.create_estimator("GaussianNB")
        model_4 = self._estimator_creator.create_estimator("LinearSVC")
        estimators = [model_1, model_2, model_3, model_4.set_params(dual=False)]
        # get score from a linearSVC estimator with accuracy score and 5folds
        bol_results = []
        for clf in estimators:
            score = cv_score.get_score(x, y, clf, "accuracy", 5)
            print(clf.__class__.__name__, "score is:", score)
            is_greater_than_zero: bool = True if score > 0 else False
            bol_results.append(is_greater_than_zero)
        # any will return True if there's any truth value in the iterable.
        print(bol_results)
        answer = all(bol_results)
        # all of this should be true
        self.assertTrue(answer)

    def test_cv_score_is_more_than_zero_with_LSVR_SVR_LASSO_SGD_r2_5(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\winequality-red.csv"
        # get df with loader creator
        scsv_type = self._loader_creator.create_loader(path, "SCSV")
        df = scsv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple a svc, knn and gnb estimator
        model_1 = self._estimator_creator.create_estimator("LinearSVR")
        model_2 = self._estimator_creator.create_estimator("SVR")
        model_3 = self._estimator_creator.create_estimator("Lasso")
        model_4 = self._estimator_creator.create_estimator("SGDClassifier")
        estimators = [model_1, model_2, model_3, model_4]
        # get score from a linearSVC estimator with accuracy score and 5folds
        bol_results = []
        for clf in estimators:
            score = cv_score.get_score(x, y, clf, "r2", 5)
            print(clf.__class__.__name__, "score is:", score)
            is_greater_than_zero: bool = True if score > 0 else False
            bol_results.append(is_greater_than_zero)
        print(bol_results)
        # there is at least one true element, which means on of the scores is greater than 0
        self.assertTrue(any(bol_results))

    def test_cv_score_is_more_than_zero_with_LSVR_SVR_LASSO_SGD_explained_variance_5(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\winequality-white.csv"
        # get df with loader creator
        scsv_type = self._loader_creator.create_loader(path, "SCSV")
        df = scsv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple a svc, knn and gnb estimator
        model_1 = self._estimator_creator.create_estimator("LinearSVR")
        model_2 = self._estimator_creator.create_estimator("SVR")
        model_3 = self._estimator_creator.create_estimator("Lasso")
        model_4 = self._estimator_creator.create_estimator("SGDClassifier")
        estimators = [model_1, model_2, model_3, model_4]
        # get score from a linearSVC estimator with accuracy score and 5folds
        bol_results = []
        for clf in estimators:
            score = cv_score.get_score(x, y, clf, "explained_variance", 5)
            print(clf.__class__.__name__, "score is:", score)
            is_greater_than_zero: bool = True if score > 0 else False
            bol_results.append(is_greater_than_zero)
        print(bol_results)
        # there is at least one true element, which means on of the scores is greater than 0
        self.assertTrue(any(bol_results))

    def test_cv_score_is_more_than_zero_with_APROPAGATION_KMEANS_MINIKMEANS_MEANSHIFT_mutual_info_score_5(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\iris.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple a svc, knn and gnb estimator
        model_1 = self._estimator_creator.create_estimator("AffinityPropagation")
        model_2 = self._estimator_creator.create_estimator("KMeans")
        model_3 = self._estimator_creator.create_estimator("MiniBatchKMeans")
        model_4 = self._estimator_creator.create_estimator("MeanShift")
        estimators = [model_1.set_params(random_state=0), model_2, model_3, model_4]
        # get score from a linearSVC estimator with accuracy score and 5folds
        bol_results = []
        for clf in estimators:
            score = cv_score.get_score(x, y, clf, "mutual_info_score", 5)
            print(clf.__class__.__name__, "score is:", score)
            is_greater_than_zero: bool = True if score > 0 else False
            bol_results.append(is_greater_than_zero)
        print(bol_results)


if __name__ == '__main__':
    unittest.main()
