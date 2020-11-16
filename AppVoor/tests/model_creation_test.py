import unittest

from estimator_creation import EstimatorCreator
from load_data import LoaderCreator
from model_creation import SBSModelCreator


class MyTestCase(unittest.TestCase):
    _loader_creator = LoaderCreator.get_instance()
    _model_creator = SBSModelCreator.get_instance()
    _estimator_creator = EstimatorCreator.get_instance()

    def test_simple_model_score_is_float_and_greater_than_zero(self):
        # create a simple model using SBSModelCreator
        model_instance = self._model_creator.create_model(False, False)
        # path to diabetes.csv file in project
        path = ".\\..\\datasets\\diabetes.csv"
        # get df with loader creator
        csv_type = self._loader_creator.create_loader(path, "CSV")
        df = csv_type.get_file_transformed()
        # create a prm variable to store params grid
        prm = {'C': 2, 'random_state': 0, 'tol': 0.01, "dual": False}
        # create an estimator using EstimatorCreator
        estimator = self._estimator_creator.create_estimator("LSVC")
        # set object best params and base estimator
        model_instance.best_params = prm
        model_instance.estimator = estimator
        score = model_instance.score_model(df, "roc_auc")
        is_valid = True if isinstance(score, float) and score > 0.0 else False
        self.assertTrue(is_valid)


if __name__ == '__main__':
    unittest.main()
