import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state
from numpy.random import RandomState
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


class RandomClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, n_components=100, random_state=None):
        self.random_state = random_state
        self.n_components = n_components

    def fit(self,X = None, y = None):
        self.X, self.y = X,y
        self.random_state = check_random_state(self.random_state)

    def predict(self, X, random_state=0):
        X = check_array(X)
        # random_state = check_random_state(random_state)
        random_state = check_random_state(random_state)
        i = random_state.randint(X.shape[0])
        return X[i]


X, y = datasets.make_classification(
    # n_features=2,
    n_samples=200,
    n_informative=2,
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = GaussianNB()
clf.fit(X_train,y_train)
predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)
print("Accuracy score: \n%.2f" % score)




