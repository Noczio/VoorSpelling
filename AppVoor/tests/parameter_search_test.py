import unittest

from scipy.stats import loguniform
from sklearn.svm import LinearSVC

from load_data import LoaderCreator
from parameter_search import ParamSearchCreator
from skopt.space import Real, Integer, Categorical
from split_data import SplitterReturner

import numpy as np


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _param_search_creator = ParamSearchCreator.get_instance()

    def test_diabetes_bayesian_search(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple linearSVC estimator
        model = LinearSVC(dual=False)
        # create a prm variable that stores the param grid to search
        prm = {'C': Integer(1, 10, prior='log-uniform'),
               'tol': Real(1e-4, 1e+1, prior='log-uniform'),
               'random_state': Integer(0, 10)}
        # create a ps variable that stores a bayesian search object
        ps = self._param_search_creator.create_param_selector("BS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """ 
        OrderedDict([('C', 5), ('random_state', 0), ('tol', 0.0011660530130053145)])
        OrderedDict([('C', 4), ('random_state', 2), ('tol', 0.0017481946152423605)])
        -> it changes every time
        """

    def test_diabetes_grid_search(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple linearSVC estimator
        model = LinearSVC(dual=False)
        # create a prm variable that stores the param grid to search
        prm = {'C': np.arange(1, 5, 1),
               'tol': np.arange(0.01, 1, 0.1),
               'random_state': np.arange(0, 10, 1)}
        # create a ps variable that stores a grid search object
        ps = self._param_search_creator.create_param_selector("GS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """
        {'C': 2, 'random_state': 0, 'tol': 0.01}
        -> it is always the same
        """


if __name__ == '__main__':
    unittest.main()
