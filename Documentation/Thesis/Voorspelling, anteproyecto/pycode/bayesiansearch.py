from skopt import BayesSearchCV

from sklearn.datasets import load_iris
from sklearn.svm import SVC

X, y = load_iris(True)

search_cv = BayesSearchCV(
SVC(gamma='scale'),
search_spaces={'C': (0.01, 100.0, 'log-uniform')},
n_iter=10,
cv=3
)