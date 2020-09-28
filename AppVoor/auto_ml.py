from interface import implements, Interface
from supervised import AutoML

from jsonInfo.random_generator import get_random_number_range_int


class IAutoMachineLearning(Interface):

    def fit_model(self, x_train, y_train):
        pass

    def predict_model(self, x_test) -> tuple:
        pass


class JarAutoML(implements(IAutoMachineLearning)):

    def __init__(self, n_folds_validation: int, shuffle_data=False, max_rand=1234):
        # initialize a random seed number
        self._random_state = get_random_number_range_int(0, max_rand, 1)
        # initialize clf. parameters n_folds_validation, shuffle_data, random_state
        self._clf = AutoML(
            mode="Compete",
            explain_level=0,
            random_state=self._random_state,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": n_folds_validation,
                "shuffle": shuffle_data
            })

    # interface method implementation
    def fit_model(self, x_train, y_train):
        # clf fit method
        self._clf.fit(x_train, y_train)

    # interface method implementation
    def predict_model(self, x_test) -> tuple:
        # clf predict. Returns prediction as tuple
        prediction_tuple = tuple(self._clf.predict(x_test))
        return prediction_tuple
