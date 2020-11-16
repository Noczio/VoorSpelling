import unittest

from estimator_creation import EstimatorCreator
from load_data import LoaderCreator
from parameter_search import ParamSearchCreator
from skopt.space import Real, Integer, Categorical
from split_data import SplitterReturner

import numpy as np


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _param_search_creator = ParamSearchCreator.get_instance()
    _estimator_creator = EstimatorCreator.get_instance()

    def test_diabetes_LSVC_bayesian_search(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple linearSVC estimator
        model = self._estimator_creator.create_estimator("LSVC")
        # create a prm variable that stores the param grid to search
        prm = {'C': Integer(1, 10, prior='log-uniform'),
               'tol': Real(1e-4, 1e+1, prior='log-uniform'),
               'random_state': Integer(0, 10),
               "dual": (False,)}
        # create a ps variable that stores a bayesian search object
        ps = self._param_search_creator.create_param_selector("BS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """ 
        OrderedDict([('C', 5), ('random_state', 0), ('tol', 0.0011660530130053145)])
        OrderedDict([('C', 4), ('random_state', 2), ('tol', 0.0017481946152423605)])
        OrderedDict([('C', 5), ('dual', False), ('random_state', 6), ('tol', 0.0011783718838941084)])
        -> it changes every time
        """

    def test_diabetes_LSVC_grid_search(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple linearSVC estimator
        model = self._estimator_creator.create_estimator("LSVC")
        # create a prm variable that stores the param grid to search
        prm = {'C': np.arange(1, 5, 1),
               'tol': np.arange(0.01, 1, 0.1),
               'random_state': np.arange(0, 10, 1),
               "dual": (False,)}
        # create a ps variable that stores a grid search object
        ps = self._param_search_creator.create_param_selector("GS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """
        {'C': 2, 'random_state': 0, 'tol': 0.01}
        {'C': 2, 'dual': False, 'random_state': 0, 'tol': 0.01}
        -> it is always the same
        """

    def test_molecules_SVC_bayesian_search(self):
        # path to molecules.csv file in project
        path = ".\\..\\datasets\\molecules.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "TSV")
        df = csv_type.get_file_transformed()
        df = df.drop(["m_name"], axis=1)
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple SVC estimator
        model = model = self._estimator_creator.create_estimator("SVC")
        # create a prm variable that stores the param grid to search
        prm = {'C': Integer(1, 10, prior='log-uniform'),
               'tol': Real(1e-4, 1e+1, prior='log-uniform'),
               'random_state': Integer(0, 10),
               "gamma": Categorical(["scale", "auto"])}
        # create a ps variable that stores a bayesian search object
        ps = self._param_search_creator.create_param_selector("BS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """
        OrderedDict([('C', 5), ('gamma', 'auto'), ('random_state', 3), ('tol', 0.015054999513470727)])
        """

    def test_molecules_SVC_grid_search(self):
        # path to molecules.csv file in project
        path = ".\\..\\datasets\\molecules.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "TSV")
        df = csv_type.get_file_transformed()
        df = df.drop(["m_name"], axis=1)
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a simple SVC estimator
        model = model = self._estimator_creator.create_estimator("SVC")
        # create a prm variable that stores the param grid to search
        prm = {'C': np.arange(1, 10, 1),
               'tol': np.arange(0.01, 1, 0.1),
               'random_state': np.arange(0, 10, 1),
               "gamma": ("auto",)}
        # create a ps variable that stores a grid search object
        ps = self._param_search_creator.create_param_selector("GS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """
        {'C': 3, 'gamma': 'auto', 'random_state': 0, 'tol': 0.51}
        """


if __name__ == '__main__':
    unittest.main()
