import unittest

from resources.backend_scripts.estimator_creation import EstimatorCreator
from resources.backend_scripts.feature_selection import FeatureSelectorCreator
from resources.backend_scripts.load_data import LoaderCreator
from resources.backend_scripts.model_creation import SBSModelCreator
from resources.backend_scripts.parameter_search import ParameterSearchCreator


class MyTestCase(unittest.TestCase):
    _estimator_creator = EstimatorCreator()
    _parameter_search_creator = ParameterSearchCreator()
    _model_creator = SBSModelCreator()
    _feature_selection_creator = FeatureSelectorCreator()
    _loader_creator = LoaderCreator()

    def test_available_types_in_EstimatorCreator(self):
        types = self._estimator_creator.get_available_types()
        expected = ('AffinityPropagation',
                    'GaussianNB',
                    'KMeans',
                    'KNeighborsClassifier',
                    'Lasso',
                    'LinearSVC',
                    'LinearSVR',
                    'MeanShift',
                    'MiniBatchKMeans',
                    'SGDClassifier',
                    'SVC',
                    'SVR')
        results = [True for i in types if i in expected]
        bol_answer = all(results)
        self.assertTrue(bol_answer)

    def test_available_types_in_ParameterSearchCreator(self):
        types = self._parameter_search_creator.get_available_types()
        expected = ('BS',
                    'BayesianSearch',
                    'GS',
                    'GridSearch')
        results = [True for i in types if i in expected]
        bol_answer = all(results)
        self.assertTrue(bol_answer)

    def test_available_types_in_SBSModelCreator(self):
        types = self._model_creator.get_available_types()
        expected = ('AM',
                    'FSM',
                    'FeatureAndParameterSearch',
                    'OnlyFeatureSelection',
                    'OnlyParameterSearch',
                    'PSM',
                    'SM',
                    'Simple')
        results = [True for i in types if i in expected]
        bol_answer = all(results)
        self.assertTrue(bol_answer)

    def test_available_types_in_FeatureSelectorCreator(self):
        types = self._feature_selection_creator.get_available_types()
        expected = ('BFS', 'BackwardsFeatureSelection', 'FFS', 'ForwardFeatureSelection')
        results = [True for i in types if i in expected]
        bol_answer = all(results)
        self.assertTrue(bol_answer)

    def test_available_types_in_LoaderCreator(self):
        types = self._loader_creator.get_available_types()
        expected = ("CSV", "TSV", "SCSV", "JSON")
        results = [True for i in types if i in expected]
        bol_answer = all(results)
        self.assertTrue(bol_answer)


if __name__ == '__main__':
    unittest.main()
