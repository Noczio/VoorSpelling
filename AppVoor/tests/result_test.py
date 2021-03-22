import unittest

from backend_scripts.estimator_creation import EstimatorCreator
from backend_scripts.global_vars import GlobalVariables
from backend_scripts.result_creation import FCreator, SBSResult


class MyTestCase(unittest.TestCase):
    estimator_creator = EstimatorCreator.get_instance()

    def test_finishes_creation(self):
        try:
            _ = FCreator(".\\")
            self.assertTrue(True)
        except():
            self.assertTrue(False)

    def test_markdown_file_creates_console_info(self):
        lots_of_info = ["test info first paragraph", "this comes after a jump line", "",
                        "this comes after two jump lines "]
        is_finished = SBSResult.console_info(lots_of_info, ".\\SBS_ML_1")
        self.assertTrue(is_finished)

    def test_markdown_file_creates_console_info_overwrite(self):
        lots_of_info = ["test info first paragraph", "this comes after a jump line", "",
                        "this comes after two jump lines", "this is new text for overwriting purposes"]
        is_finished = SBSResult.console_info(lots_of_info, ".\\SBS_ML_1")
        self.assertTrue(is_finished)

    def test_markdown_file_creates_estimator_info(self):
        global_var = GlobalVariables.get_instance()
        global_var.estimator = self.estimator_creator.create_estimator("KNeighborsClassifier")
        global_var.prediction_type = "Classification"
        global_var.uses_feature_selection = False
        global_var.uses_parameter_search = True
        global_var.parameters = {"k": 1, "p": 1}
        initial_parameters = global_var.parameters
        best_parameters = {"k": 3, "p": 2}
        score_text = "rendimiento promedio ROC_AUC:" + " " + str(0.7886)
        best_features = ["dryness", "elevation", "temeperature", "arrival_time", "has_objects", "n_of_near_species"]
        info = ["Opción", "Selección",
                "Tipo de predicción", global_var.prediction_type,
                "Estimador", global_var.estimator.__class__.__name__,
                "Selección de características", str(global_var.uses_feature_selection),
                "Selección de hiperparámetros", str(global_var.uses_parameter_search),
                ]
        table = {"columns": 2, "rows": 5, "info": info}
        folder_path = ".\\SBS_ML_1"
        is_finished = SBSResult.estimator_info(table,
                                               list(best_features),
                                               initial_parameters,
                                               best_parameters,
                                               score_text,
                                               folder_path)

        self.assertTrue(is_finished)


if __name__ == '__main__':
    unittest.main()
