import unittest

from estimator_creation import EstimatorCreator
from load_data import LoaderCreator
from parameter_search import ParameterSearchCreator
from parameter_search import BayesianSearchParametersPossibilities, GridSearchParametersPossibilities
from skopt.space import Real, Integer, Categorical
from split_data import SplitterReturner

import numpy as np


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _param_search_creator = ParameterSearchCreator.get_instance()
    _estimator_creator = EstimatorCreator.get_instance()

    def test_molecules_SVC_bayesian_search(self):
        # path to molecules.csv file in project
        path = ".\\..\\datasets\\molecules.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "TSV")
        df = csv_type.get_file_transformed()
        df = df.drop(["m_name"], axis=1)
        # split df into x and y
        x, y = SplitterReturner.split_x_y_from_df(df)
        # create a simple SVC estimator
        model = self._estimator_creator.create_estimator("SVC")
        # create a prm variable that stores the param grid to search
        prm = BayesianSearchParametersPossibilities.case("SVC")
        # create a ps variable that stores a bayesian search object
        ps = self._param_search_creator.create_parameter_selector("BS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """
        
        """

    def test_wine_quality_LASSO_BS(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\winequality-red.csv"
        # get df with loader creator
        scsv_type = self._loader_creator.create_loader(path, "SCSV")
        df = scsv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = BayesianSearchParametersPossibilities.case("Lasso")
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("Lasso")
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a ps variable that stores a grid search object
        ps = self._param_search_creator.create_parameter_selector("BS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, initial_prm, 10, estimator)
        print(best_prm)
        """
        OrderedDict([('alpha', 1.0), ('positive', False), ('selection', 'random'), ('tol', 0.0001)])
        """

    def test_diabetes_lsvc_search_bs(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # split df into x and y
        x, y = SplitterReturner.split_x_y_from_df(df)
        # create a simple linearSVC estimator
        model = self._estimator_creator.create_estimator("LinearSVC")
        # create a prm variable that stores the param grid to search
        prm = BayesianSearchParametersPossibilities.case("LinearSVC")
        # create a ps variable that stores a bayesian search object
        ps = self._param_search_creator.create_parameter_selector("BS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """
        OrderedDict([('C', 1.2042353730174158), ('dual', False), ('intercept_scaling', 1.0), ('penalty', 'l2'),
         ('tol', 0.0001)])
        """

    def test_wine_quality_LASSO_GS(self):
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\winequality-white.csv"
        # get df with loader creator
        scsv_type = self._loader_creator.create_loader(path, "SCSV")
        df = scsv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'alpha': np.arange(1, 10, 0.5),
                       'tol': np.arange(1e-4, 1e+1, 0.1),
                       'random_state': np.arange(0, 10, 1),
                       "selection": ("cyclic", "random")}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("Lasso")
        # split df into x and y
        splitter = SplitterReturner()
        x, y = splitter.split_x_y_from_df(df)
        # create a ps variable that stores a grid search object
        ps = self._param_search_creator.create_parameter_selector("GS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, initial_prm, 10, estimator)
        print(best_prm)
        """

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
        model = self._estimator_creator.create_estimator("SVC")
        # create a prm variable that stores the param grid to search
        prm = {'C': np.arange(1, 10, 1),
               'tol': np.arange(0.01, 1, 0.1),
               'random_state': np.arange(0, 10, 1),
               "gamma": ("auto",)}
        # create a ps variable that stores a grid search object
        ps = self._param_search_creator.create_parameter_selector("GS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """

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
        model = self._estimator_creator.create_estimator("LinearSVC")
        # create a prm variable that stores the param grid to search
        prm = {'C': np.arange(1, 5, 1),
               'tol': np.arange(0.01, 1, 0.1),
               'random_state': np.arange(0, 10, 1),
               "dual": (False,)}
        # create a ps variable that stores a grid search object
        ps = self._param_search_creator.create_parameter_selector("GS")
        # get best params from ps.search_parameters
        best_prm = ps.search_parameters(x, y, prm, 10, model)
        print(best_prm)
        """

        """


if __name__ == '__main__':
    unittest.main()
