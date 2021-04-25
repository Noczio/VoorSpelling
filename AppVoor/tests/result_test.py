import sys
import unittest
from io import StringIO

from backend_scripts.estimator_creation import EstimatorCreator
from backend_scripts.feature_selection import FeatureSelectorCreator
from backend_scripts.load_data import LoaderCreator
from backend_scripts.model_creation import SBSModelCreator
from backend_scripts.parameter_search import ParameterSearchCreator, BayesianSearchParametersPossibilities
from backend_scripts.result_creation import FCreator, SBSResult


class MyTestCase(unittest.TestCase):
    _estimator_creator = EstimatorCreator.get_instance()
    _loader_creator = LoaderCreator.get_instance()
    _model_creator = SBSModelCreator.get_instance()
    _feature_selection_creator = FeatureSelectorCreator.get_instance()
    _parameter_selection_creator = ParameterSearchCreator.get_instance()

    def test_finishes_creation(self):
        try:
            _ = FCreator(".\\")
            self.assertTrue(True)
        except():
            self.assertTrue(False)

    def test_markdown_file_creates_console_info(self):
        lots_of_info = ["test info first paragraph", "this comes after a jump line", "",
                        "this comes after two jump lines "]
        SBSResult.console_info(lots_of_info, ".\\SBS_ML_1")

    def test_markdown_file_creates_console_info_overwrite(self):
        lots_of_info = ["test info first paragraph", "this comes after a jump line", "",
                        "this comes after two jump lines", "this is new text for overwriting purposes"]
        SBSResult.console_info(lots_of_info, ".\\SBS_ML_1")

    def test_markdown_file_creates_estimator_info(self):
        file_creator_obj = FCreator()
        uses_parameter_search, uses_feature_selection = True, False
        model_instance = self._model_creator.create_model(uses_feature_selection, uses_parameter_search)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("KNeighborsClassifier")
        # set model instance attributes
        model_instance.initial_parameters = BayesianSearchParametersPossibilities.case("KNeighborsClassifier")
        model_instance.estimator = estimator
        model_instance.parameter_selector = self._parameter_selection_creator.create_parameter_selector("BS")
        score = model_instance.score_model(df, "accuracy", 10)
        print("score:", score)
        print("best params", model_instance.best_parameters)
        print("best features", model_instance.best_features)
        score_text = "rendimiento promedio \"accuracy\":" + " " + str(score)

        info = ["Opción", "Selección",
                "Tipo de predicción", "Classification",
                "Estimador", model_instance.estimator.__class__.__name__,
                "Selección de características", "No" if model_instance.feature_selector is None
                else model_instance.feature_selector.__class__.__name__,
                "Selección de hiperparámetros", "No" if model_instance.parameter_selector is None
                else model_instance.parameter_selector.__class__.__name__
                ]
        table = {"columns": 2, "rows": 5, "info": info}
        folder_path = file_creator_obj.folder_path
        SBSResult.estimator_info(table,
                                 list(model_instance.best_features),
                                 model_instance.initial_parameters,
                                 model_instance.best_parameters,
                                 score_text,
                                 folder_path)
        """
        score: 0.7604166666666666
        best params OrderedDict([('algorithm', 'kd_tree'), ('leaf_size', 30), ('n_neighbors', 13), ('p', 1),
         ('weights', 'uniform')])
        best features ['Pregnancies' 'Glucose' 'BloodPressure' 'SkinThickness' 'Insulin' 'BMI'
         'DiabetesPedigreeFunction' 'Age']
        """

    def test_markdown_file_creates_estimator_and_console_info(self):
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        file_creator_obj = FCreator()
        uses_parameter_search, uses_feature_selection = True, True
        model_instance = self._model_creator.create_model(uses_feature_selection, uses_parameter_search)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("KNeighborsClassifier")
        # set model instance attributes
        model_instance.initial_parameters = BayesianSearchParametersPossibilities.case("KNeighborsClassifier")
        model_instance.estimator = estimator
        model_instance.parameter_selector = self._parameter_selection_creator.create_parameter_selector("BS")
        model_instance.feature_selector = self._feature_selection_creator.create_feature_selector("BFS")
        score = model_instance.score_model(df, "accuracy", 10)
        score_text = "rendimiento promedio \"accuracy\":" + " " + str(score)

        info = ["Opción", "Selección",
                "Tipo de predicción", "Clasificación",
                "Estimador", model_instance.estimator.__class__.__name__,
                "Selección de características", "No" if model_instance.feature_selector is None
                else model_instance.feature_selector.__class__.__name__,
                "Selección de hiperparámetros", "No" if model_instance.parameter_selector is None
                else model_instance.parameter_selector.__class__.__name__
                ]
        table = {"columns": 2, "rows": 5, "info": info}
        folder_path = file_creator_obj.folder_path
        SBSResult.estimator_info(table,
                                 list(model_instance.best_features),
                                 model_instance.initial_parameters,
                                 model_instance.best_parameters,
                                 score_text,
                                 folder_path)

        sys.stdout = old_stdout
        console_output = my_stdout.getvalue()
        SBSResult.console_info(console_output.split("\n"), folder_path, 0)

    def test_markdown_file_creates_estimator_and_console_info_with_wine_quality_red_sgd_bfs(self):
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        file_creator_obj = FCreator()
        uses_parameter_search, uses_feature_selection = False, True
        model_instance = self._model_creator.create_model(uses_feature_selection, uses_parameter_search)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\winequality-red-coma.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SGDClassifier")
        # set model instance attributes
        model_instance.initial_parameters = {'penalty': 'elasticnet',
                                             'alpha': 0.01,
                                             'tol': 0.0001,
                                             'random_state': 512}
        model_instance.estimator = estimator
        model_instance.feature_selector = self._feature_selection_creator.create_feature_selector("BFS")
        score = model_instance.score_model(df, "neg_mean_squared_error", 10)
        score_text = "rendimiento promedio \"neg_mean_squared_error\":" + " " + str(score)

        info = ["Opción", "Selección",
                "Tipo de predicción", "Clasificación",
                "Estimador", model_instance.estimator.__class__.__name__,
                "Selección de características", "No" if model_instance.feature_selector is None
                else model_instance.feature_selector.__class__.__name__,
                "Selección de hiperparámetros", "No" if model_instance.parameter_selector is None
                else model_instance.parameter_selector.__class__.__name__
                ]
        table = {"columns": 2, "rows": 5, "info": info}
        folder_path = file_creator_obj.folder_path
        SBSResult.estimator_info(table,
                                 list(model_instance.best_features),
                                 model_instance.initial_parameters,
                                 model_instance.best_parameters,
                                 score_text,
                                 folder_path)

        sys.stdout = old_stdout
        console_output = my_stdout.getvalue()
        SBSResult.console_info(console_output.split("\n"), folder_path, 0)

    def test_markdown_file_dumps_estimator_and_creates_estimator_and_console_info_with_wine_quality_red_sgd_ffs(self):
        old_stdout = sys.stdout
        sys.stdout = my_stdout = StringIO()
        file_creator_obj = FCreator()
        uses_parameter_search, uses_feature_selection = False, True
        model_instance = self._model_creator.create_model(uses_feature_selection, uses_parameter_search)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\winequality-red-coma.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("SGDClassifier")
        # set model instance attributes
        model_instance.initial_parameters = {'penalty': 'elasticnet',
                                             'alpha': 0.01,
                                             'tol': 0.0001,
                                             'random_state': 512}
        model_instance.estimator = estimator
        model_instance.feature_selector = self._feature_selection_creator.create_feature_selector("FFS")
        score = model_instance.score_model(df, "neg_mean_squared_error", 10)
        score_text = "rendimiento promedio \"neg_mean_squared_error\":" + " " + str(score)

        info = ["Opción", "Selección",
                "Tipo de predicción", "Clasificación",
                "Estimador", model_instance.estimator.__class__.__name__,
                "Selección de características", "No" if model_instance.feature_selector is None
                else model_instance.feature_selector.__class__.__name__,
                "Selección de hiperparámetros", "No" if model_instance.parameter_selector is None
                else model_instance.parameter_selector.__class__.__name__
                ]
        table = {"columns": 2, "rows": 5, "info": info}
        folder_path = file_creator_obj.folder_path
        SBSResult.estimator_info(table,
                                 list(model_instance.best_features),
                                 model_instance.initial_parameters,
                                 model_instance.best_parameters,
                                 score_text,
                                 folder_path)
        SBSResult.dump_estimator(estimator, folder_path)
        sys.stdout = old_stdout
        console_output = my_stdout.getvalue()
        SBSResult.console_info(console_output.split("\n"), folder_path, 0)


if __name__ == '__main__':
    unittest.main()
