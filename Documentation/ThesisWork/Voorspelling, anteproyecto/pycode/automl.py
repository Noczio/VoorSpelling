from sklearn.datasets import load_iris
from supervised import AutoML


X, y = load_iris(True)
clf = AutoML(
            mode="Compete",
            explain_level=0,
            random_state=0,
            validation_strategy={
                "validation_type": "kfold",
                "k_folds": 10,
                "shuffle": False
            })
clf.fit(X,y)