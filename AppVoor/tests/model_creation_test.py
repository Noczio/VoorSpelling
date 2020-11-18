import unittest

from estimator_creation import EstimatorCreator
from feature_selection import FeatureSelectorCreator
from is_data import DataEnsurer
from load_data import LoaderCreator
from model_creation import SBSModelCreator

from skopt.space import Real, Integer, Categorical

from parameter_search import ParamSearchCreator


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _model_creator = SBSModelCreator.get_instance()
    _estimator_creator = EstimatorCreator.get_instance()
    _feature_selection_creator = FeatureSelectorCreator.get_instance()
    _parameter_selection_creator = ParamSearchCreator.get_instance()

    def test_simple_model_LSVC_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'random_state': 0, 'tol': 0.01, "dual": False}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("LSVC")
        # set object best params and base estimator
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if isinstance(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_simple_model_SVC_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'random_state': 0, 'tol': 0.01, "kernel": "rbf"}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # set object best params and base estimator
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_feature_selection_model_SVC_FFS_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'random_state': 0, 'tol': 0.01, "kernel": "rbf"}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("FFS")
        # set object best params, base estimator and feature selector
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_feature_selection_model_SVC_SFM_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'random_state': 0, 'tol': 0.01, "kernel": "rbf"}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("SFM")
        # set object best params, base estimator and feature selector
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_feature_selection_model_SVC_BFS_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'random_state': 0, 'tol': 0.01, "kernel": "rbf"}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("BFS")
        # set object best params, base estimator and feature selector
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_feature_selection_model_LSVC_RFE_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': 2, 'random_state': 0, 'tol': 0.01, "dual": False}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("LSVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("RFE")
        # set object best params, base estimator and feature selector
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_only_parameter_search_model_SVC_BS_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, True)
        # path to molecules.csv file in project
        path = ".\\..\\datasets\\molecules.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "TSV")
        df = csv_type.get_file_transformed()
        df = df.drop(["m_name"], axis=1)
        # create a prm variable to store params grid
        initial_prm = {'C': Integer(1, 10, prior='log-uniform'),
                       'tol': Real(1e-4, 1e+1, prior='log-uniform'),
                       'random_state': Integer(0, 10),
                       "gamma": Categorical(["scale", "auto"])}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a parameter selector variable to store a ParameterSearch instance
        parameter_selector = self._parameter_selection_creator.create_param_selector("BS")
        # set object best params, base estimator and parameter selector
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        model_instance.parameter_selector = parameter_selector
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)

    def test_all_model_SVC_BS_FFM_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(True, True)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        initial_prm = {'C': Integer(1, 10, prior='log-uniform'),
                       'tol': Real(1e-4, 1e+1, prior='log-uniform'),
                       'random_state': Integer(0, 10),
                       "gamma": Categorical(["scale", "auto"])}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SVC")
        # create a feature selector variable to store a FeatureSelection instance
        feature_selector = self._feature_selection_creator.create_feature_selector("FFS")
        # create a parameter selector variable to store a ParameterSearch instance
        parameter_selector = self._parameter_selection_creator.create_param_selector("BS")
        # set object best params, base estimator, parameter selector and feature selector
        model_instance.initial_params = initial_prm
        model_instance.estimator = estimator
        model_instance.feature_selector = feature_selector
        model_instance.parameter_selector = parameter_selector
        score = model_instance.score_model(df, "roc_auc")
        print("score:", score)
        print("best params", model_instance.best_params)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)


if __name__ == '__main__':
    unittest.main()
