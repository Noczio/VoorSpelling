from abc import ABC, abstractmethod

from supervised import AutoML

from jsonInfo.random_generator import get_random_number_range_int


class ABCAutoMachineLearning(ABC):

    def __init__(self, n_folds_validation: int, shuffle_data: bool, max_rand: int) -> None:
        # initialize ._random_state, _n_folds_validation and _shuffle_data
        self._random_state = get_random_number_range_int(0, max_rand, 1)
        self._n_folds_validation = n_folds_validation
        self._shuffle_data = shuffle_data

    @abstractmethod
    def fit_model(self, x_train, y_train) -> None:
        pass

    @abstractmethod
    def predict_model(self, x_test) -> tuple:
        pass


class JarAutoML(ABCAutoMachineLearning):

    def __init__(self, n_folds_validation: int, shuffle_data: bool, max_rand: int) -> None:
        super().__init__(n_folds_validation, shuffle_data, max_rand)
        # if by any case the random state state is fewer than 0 then fix it to 0
        # var  = [false,true][test]
        self._random_state = [self._random_state, 0][self._random_state < 0]
        self._clf = AutoML(
            mode="Compete",
            explain_level=0,
            random_state=self._random_state,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": self._n_folds_validation,
                "shuffle": self._shuffle_data
            })

    # abstract class method implementation
    def fit_model(self, x_train, y_train) -> None:
        # clf fit method
        self._clf.fit(x_train, y_train)

    # abstract class method implementation
    def predict_model(self, x_test) -> tuple:
        # clf predict. Returns prediction as tuple
        prediction_tuple = tuple(self._clf.predict(x_test))
        return prediction_tuple
