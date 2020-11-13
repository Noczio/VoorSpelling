import unittest

from scipy.stats import loguniform
from sklearn.svm import LinearSVC

from load_data import LoaderCreator
from parameter_search import BayesianSearch, GridSearch
from skopt.space import Real, Integer, Categorical
from split_data import SplitterReturner

import numpy as np


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()

    def test_diabetes_bayesian_search(self):
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(".\\..\\datasets\\diabetes.csv", "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple linearSVC estimator
        model = LinearSVC(dual=False)
        prm = {'C': Integer(1, 10, prior='log-uniform'),
               'tol': Real(1e-4, 1e+1, prior='log-uniform'),
               'random_state': Integer(0, 10)}

        ps = BayesianSearch()
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """ 
        OrderedDict([('C', 5), ('random_state', 0), ('tol', 0.0011660530130053145)])
        OrderedDict([('C', 4), ('random_state', 2), ('tol', 0.0017481946152423605)])
        """

    def test_diabetes_grid_search(self):
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(".\\..\\datasets\\diabetes.csv", "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple linearSVC estimator
        model = LinearSVC(dual=False)
        prm = {'C': loguniform(1, 5),
               'tol': loguniform(1e-2, 1e+1),
               'random_state': loguniform(0, 10)}
        ps = GridSearch()
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)


if __name__ == '__main__':
    unittest.main()
