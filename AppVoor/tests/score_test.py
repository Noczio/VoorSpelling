import unittest

from load_data import LoaderCreator
from score import CVScore
from split_data import SplitterReturner
from sklearn.svm import LinearSVC


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()

    def test_score_is_more_than_zero(self):
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(".\\..\\datasets\\diabetes.csv", "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple linearSVC estimator
        model = LinearSVC(random_state=0, dual=False)
        # get score from a linearSVC estimator with roc_auc score and 10 folds
        score = cv_score.get_score(x, y, model, "roc_auc", 10)
        is_greater_than_zero: bool = True if score > 0 else False
        self.assertTrue(is_greater_than_zero)

    def test_score_type_raises_ValueError(self):
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(".\\..\\datasets\\diabetes.csv", "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple linearSVC estimator
        model = LinearSVC(random_state=0, dual=False)
        with self.assertRaises(ValueError):
            # get score from a linearSVC estimator with roc_auc score and 10 folds
            _ = cv_score.get_score(x, y, model, "roc", 10)

    def test_n_folds_validation_raises_ValueError(self):
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(".\\..\\datasets\\diabetes.csv", "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple linearSVC estimator
        model = LinearSVC(random_state=0, dual=False)
        with self.assertRaises(ValueError):
            # get score from a linearSVC estimator with roc_auc score and 10 folds
            _ = cv_score.get_score(x, y, model, "roc_auc", 2)

    def test_n_folds_validation_and_score_type_raises_ValueError(self):
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(".\\..\\datasets\\diabetes.csv", "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a CVScore object with its path and data type
        cv_score = CVScore()
        # create a simple linearSVC estimator
        model = LinearSVC(random_state=0, dual=False)
        with self.assertRaises(ValueError):
            # get score from a linearSVC estimator with roc_auc score and 10 folds
            _ = cv_score.get_score(x, y, model, "roc", 2)


if __name__ == '__main__':
    unittest.main()
