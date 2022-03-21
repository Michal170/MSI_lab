import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state
from numpy.random import RandomState
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist



class RandomClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, n_components=100, random_state=None):
        self.random_state = random_state
        self.n_components = n_components

    def fit(self,X , y ):
        self.X, self.y = X,y
        self.random_state = check_random_state(self.random_state)

    def predict(self, X, random_state=0):
        X = check_array(X)
        random_state = check_random_state(random_state)
        i = random_state.randint(X.shape[0])
        return X[i]


X, y = datasets.make_classification(
    n_features=2,  # liczba atrybutów zbioru
    n_samples=200,  # liczba generowanych wzorców
    n_informative=2,  # liczba atrybutów informatywnych, tych które zawierają informacje przydatne dla klasyfikacji
    n_repeated=0,  # liczba atrybutów powtórzonych, czyli zduplikowanych kolumn
    n_redundant=0,  # liczba atrybutów nadmiarowych
    random_state=1410,  # ziarno losowości, pozwala na wygenerowanie dokładnie tego samego zbioru w każdym powtórzeniu
    n_classes=2,  # liczba klas problemu
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = GaussianNB()
clf.fit(X_train,y_train)
predict = clf.predict(X_test)
score = accuracy_score(y_test, predict)
print("Accuracy score: \n%.2f" % score)


############################################ZAD 2.2 ####################################################################
# cdist = cdist(X_train, y_train, metric='euclidean' )
# # cdist.fit(X_train,y_train)
# #
# cdist_predict = cdist.predict(X_test)
# cdist_score = accuracy_score(y_test, cdist_predict)


KNeighbors = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
KNeighbors.fit(X_train,y_train)


neigh_predict = KNeighbors.predict(X_test)
KNeighbors_score = accuracy_score(y_test, neigh_predict)

print("Accuracy KNeighbors_score: \n%.2f" % KNeighbors_score)
# print("Accuracy cdist score: \n%.3f" % cdist_score)


#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
#
# clf = GaussianNB()
# clf.fit(X_train,y_train)
# predict = clf.predict(X_test)
# score = accuracy_score(y_test, predict)
# print("Accuracy score: \n%.2f zad2.2" % score)