from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.svm import SVC


X, y = load_iris(True)
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC()

search_cv = GridSearchCV(svc, parameters)

