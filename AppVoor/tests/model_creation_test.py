import unittest

from estimator_creation import EstimatorCreator
from feature_selection import FeatureSelectorCreator
from is_data import DataEnsurer
from load_data import LoaderCreator
from model_creation import SBSModelCreator
from parameter_search import ParameterSearchCreator
from parameter_search import BayesianSearchParametersPossibilities, GridSearchParametersPossibilities


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _model_creator = SBSModelCreator.get_instance()
    _estimator_creator = EstimatorCreator.get_instance()
    _feature_selection_creator = FeatureSelectorCreator.get_instance()
    _parameter_selection_creator = ParameterSearchCreator.get_instance()

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
        score = model_instance.score_model(df, "roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if isinstance(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        score: 0.8301623931623933
        best params {'C': 2, 'tol': 0.01, 'dual': False, 'penalty': 'l1', 'intercept_scaling': 3.45}
        best features ['Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI'
         'DiabetesPedigreeFunction' 'Age']
        """

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
        score = model_instance.score_model(df, "roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        score: 0.5
        best params {'C': 2, 'gamma': 'auto', 'tol': 0.01, 'kernel': 'sigmoid'}
        best features ['Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI'
         'DiabetesPedigreeFunction' 'Age']
        """

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
        score = model_instance.score_model(df, "roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        best params {'C': 5, 'gamma': 'scale', 'tol': 0.01, 'kernel': 'poly'}
        best features ['Glucose' 'BMI' 'Age' 'DiabetesPedigreeFunction' 'BloodPressure'
         'Pregnancies']
        """

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
        score = model_instance.score_model(df, "roc_auc", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        score: 0.5102361984626136
        best params {'C': 5, 'gamma': 'auto', 'tol': 1, 'kernel': 'sigmoid'}
        best features ['Pregnancies' 'Glucose' 'BloodPressure']
        """

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
        score = model_instance.score_model(df, "roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        best params {'C': 2, 'gamma': 'auto', 'kernel': 'rbf', 'tol': 1.5}
        best features ['n_atoms_without_Hydrogen' 'n_atoms_with_Hydrogen' 'm_weight'
         'm_avg_weigth' 'm_weigth_without_Hydrogen' 'n_valence_electrons'
         'n_aliphatic_carbocycles' 'n_aliphatic_heterocycles' 'n_aliphatic_rings'
         'n_amide_bonds' 'n_aromatic_carbocycles' 'n_aromatic_heterocycles'
         'n_aromatic_rings' 'n_saturated_carbocycles' 'n_saturated_heterocycles'
         'n_saturated_rings' 'n_HBA' 'n_HBD' 'n_hetero_atoms' 'n_hetero_cycles'
         'n_rings' 'n_strict_rotable_bonds' 'n_non_strict_rotable_bonds'
         'n_primary_carbon_atoms' 'n_HOH' 'n_O' 'n_briged_head_atoms'
         'n_atoms_stereo_centers' 'n_atoms_unspecified_stereo_centers' 'm_logp'
         'm_mr' 'fraction_CSP3']
        """

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
        score = model_instance.score_model(df, "roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        score: 0.8299343116701609
        best params OrderedDict([('C', 7.1096025677305486), ('gamma', 'scale'), ('kernel', 'rbf'), ('tol', 1.0)])
        best features ['Glucose' 'Age' 'BMI' 'DiabetesPedigreeFunction']
        """

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
        score = model_instance.score_model(df, "r2", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) else False
        self.assertTrue(is_valid)
        """
        score: -0.0419944898180886
        best params OrderedDict([('alpha', 1.0), ('positive', False), ('selection', 'cyclic'), ('tol', 0.0001)])
        best features ['fixed acidity' 'volatile acidity' 'citric acid' 'residual sugar'
         'chlorides' 'free sulfur dioxide' 'total sulfur dioxide' 'density' 'pH'
         'sulphates' 'alcohol']
        """

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
        score = model_instance.score_model(df, "roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        score: 0.8318259958071279
        best params OrderedDict([('var_smoothing', 7.677692005912027e-05)])
        best features ['Glucose' 'BMI' 'Age' 'DiabetesPedigreeFunction']
        """

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
        score = model_instance.score_model(df, "roc_auc", 5)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        is_valid = True if DataEnsurer.validate_py_data(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)
        """
        score: 0.8144919636617749
        best params {'var_smoothing': 0.001}
        best features ['Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI'
         'DiabetesPedigreeFunction' 'Age']
        """


if __name__ == '__main__':
    unittest.main()
