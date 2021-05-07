import unittest

from resources.backend_scripts.estimator_creation import EstimatorCreator
from resources.backend_scripts.feature_selection import FeatureSelectorCreator
from resources.backend_scripts.is_data import DataEnsurer
from resources.backend_scripts.load_data import LoaderCreator
from resources.backend_scripts.model_creation import SBSModelCreator
from resources.backend_scripts.parameter_search import ParameterSearchCreator
from resources.backend_scripts.parameter_search import BayesianSearchParametersPossibilities
from resources.backend_scripts.parameter_search import GridSearchParametersPossibilities


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator()
    _model_creator = SBSModelCreator()
    _estimator_creator = EstimatorCreator()
    _feature_selection_creator = FeatureSelectorCreator()
    _parameter_selection_creator = ParameterSearchCreator()

    def test_parameters_are_wrong_raises_type_error(self):
        with self.assertRaises(TypeError):
            _ = self._model_creator.create_model("False", False)

    def test_simple_model_LSVC_roc_auc_10_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'tol': 0.01, "dual": False, 'penalty': 'l1',
                       'intercept_scaling': 3.45}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("LinearSVC")
        # set object best params and base estimator
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if isinstance(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_simple_model_SVC_roc_auc_10_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'gamma': 'auto', 'tol': 0.01, "kernel": "sigmoid"}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # set object best params and base estimator
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_feature_selection_model_SVC_FFS_roc_auc_10_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 5, 'gamma': 'scale', 'tol': 0.01, "kernel": "poly"}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("FFS")
        # set object best params, base estimator and feature selector
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_feature_selection_model_SVC_BFS__roc_auc_10_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 3, 'gamma': 'scale', 'tol': 0.0001, "kernel": "sigmoid"}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("BFS")
        # set object best params, base estimator and feature selector
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_parameter_search_model_SVC_GS_roc_auc_5_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, True)
        # path to molecules.csv file in project
        path = ".\\..\\datasets\\molecules.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "TSV")
        df = csv_type.get_file_transformed()
        df = df.drop(["m_name"], axis=1)
        # create a prm variable to store params grid
        initial_prm = GridSearchParametersPossibilities.case("SVC")
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a parameter selector variable to store a ParameterSearch instance
        parameter_selector = self._parameter_selection_creator.create_parameter_selector("GS")
        # set object best params, base estimator and parameter selector
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.parameter_selector = parameter_selector
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_all_model_SVC_BS_FFS_roc_auc_5_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, True)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = BayesianSearchParametersPossibilities.case("SVC")
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("FFS")
        # create a parameter selector variable to store a ParameterSearch instance
        parameter_selector = self._parameter_selection_creator.create_parameter_selector("BS")
        # set object best params, base estimator, parameter selector and feature selector
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        model_instance.parameter_selector = parameter_selector
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_all_model_LASSO_BS_BFS_r2_5_score_is_float(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, True)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\winequality-red.csv"
        # get df with loader creator
        scsv_type = self._loader_creator.create_loader(path, "SCSV")
        df = scsv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = BayesianSearchParametersPossibilities.case("Lasso")
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("Lasso")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("BFS")
        # create a parameter selector variable to store a ParameterSearch instance
        parameter_selector = self._parameter_selection_creator.create_parameter_selector("BS")
        # set object best params, base estimator, parameter selector and feature selector
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        model_instance.parameter_selector = parameter_selector
        model_instance.data_frame = df
        score = model_instance.score_model("r2", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) else False
        self.assertTrue(is_valid)

    def test_all_model_GNB_BS_FFS_roc_auc_5_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, True)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = BayesianSearchParametersPossibilities.case("GaussianNB")
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("GaussianNB")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("FFS")
        # create a parameter selector variable to store a ParameterSearch instance
        parameter_selector = self._parameter_selection_creator.create_parameter_selector("BS")
        # set object best params, base estimator, parameter selector and feature selector
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        model_instance.parameter_selector = parameter_selector
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_PS_model_GNB_GS_roc_auc_5_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, True)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = GridSearchParametersPossibilities.case("GaussianNB")
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("GaussianNB")
        # create a parameter selector variable to store a ParameterSearch instance
        parameter_selector = self._parameter_selection_creator.create_parameter_selector("GS")
        # set object best params, base estimator, parameter selector and feature selector
        model_instance.initial_parameters = initial_prm
        model_instance.estimator = estimator
        model_instance.parameter_selector = parameter_selector
        model_instance.data_frame = df
        score = model_instance.score_model("roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)


if __name__ == '__main__':
    unittest.main()
