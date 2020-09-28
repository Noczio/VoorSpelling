from interface import implements, Interface
from supervised import AutoML


class IAutoMachineLearning(Interface):

    def fit_model(self, x_train, y_train):
        pass

    def predict_model(self, x_test) -> tuple:
        pass


class JarAutoML(implements(IAutoMachineLearning)):

    def __init__(self, n_folds: int, shuffle_data=False):
        self._clf = AutoML(
            explain_level=0,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": n_folds,
                "shuffle": shuffle_data
            })

    def fit_model(self, x_train, y_train):
        self._clf.fit(x_train, y_train)

    def predict_model(self, x_test) -> tuple:
        prediction_tuple = tuple(self._clf.predict(x_test))
        return prediction_tuple
