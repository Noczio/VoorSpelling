import unittest

from estimator_creation import EstimatorCreator
from result_creation import FCreator, SBSResult


class MyTestCase(unittest.TestCase):
    estimator_creator = EstimatorCreator.get_instance()

    def test_finishes_creation(self):
        try:
            _ = FCreator(".\\")
            self.assertTrue(True)
        except():
            self.assertTrue(False)

    def test_markdown_file_creates_console_info(self):
        try:
            lots_of_info = "test info first paragraph\nthis comes after a jump line\n\nthis comes after two jump lines"
            SBSResult.console_info(lots_of_info, ".\\SBS_ML_1")
            self.assertTrue(True)
        except():
            self.assertTrue(False)

    def test_markdown_file_creates_console_info_overwrite(self):
        try:
            lots_of_info = "test info first paragraph\n\nthis comes after a jump line\n\nthis comes after two jump " \
                           "lines "
            SBSResult.console_info(lots_of_info, ".\\SBS_ML_1")
            self.assertTrue(True)
        except():
            self.assertTrue(False)

    def test_markdown_file_creates_estimator_info(self):
        try:
            estimator = self.estimator_creator.create_estimator("KNeighborsClassifier")
            initial_params = {"k": 1, "p": 1}
            final_params = {"k": 3, "p": 5}
            performance = "rendimiento promedio ROC_AUC:" + " " + str(0.7886)
            features = ["dryness", "elevation", "temeperature", "arrival_time", "has_objects", "n_of_near_species"]
            info = ["Opción", "Selección",
                    "Tipo de predicción", "Clasificación",
                    "Estimador", estimator.__class__.__name__,
                    "Selección de características", str(False),
                    "Selección de hiperparámetros", str(True),
                    "Tipo de entrenamiento", "Paso a paso"]
            options = {"columns": 2, "rows": 6, "info": info}
            path = ".\\SBS_ML_1"
            SBSResult.estimator_info(options, features, initial_params, final_params, performance, path)
            self.assertTrue(True)
        except():
            self.assertTrue(False)


if __name__ == '__main__':
    unittest.main()
