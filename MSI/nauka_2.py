import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted, check_X_y,check_array
from numpy.random import RandomState
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from sklearn.base import clone
from sklearn.datasets import make_classification
from numpy.random import RandomState, choice
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.multiclass import unique_labels
from prettytable import PrettyTable

############################################ZAD 2.1 ####################################################################

class RandomClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,  random_state = 1234):
        self.random_state = np.random.RandomState()
        # np.random.seed(self.random_state)

    def fit(self,X , y ):
        X,y = check_X_y(X,y)
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X,y
        self.n_classes = len(np.unique(y))
        temp = []


        for i in range(0, len(np.unique(y))):
            pointer = 0
            for j in range(0,len(y)):
                if i == y[j]:
                    pointer+=1
            temp.append(pointer)
        self.probability_ = []

        for i in temp:
            self.probability_.append((i/len(y)))

        return self


    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        y_predict = []

        for i in range(0, len(X)):
            ran_y = np.random.choice(self.classes_,1,p=self.probability_)
            y_predict.append(ran_y)

        return np.array(y_predict)




X, y = make_classification(
        n_features=2,  # liczba atrybutów zbioru
        n_samples=200,  # liczba generowanych wzorców
        n_informative=2,  # liczba atrybutów informatywnych, tych które zawierają informacje przydatne dla klasyfikacji
        n_repeated=0,  # liczba atrybutów powtórzonych, czyli zduplikowanych kolumn
        n_redundant=0,  # liczba atrybutów nadmiarowych
        # random_state=1410,
        # ziarno losowości, pozwala na wygenerowanie dokładnie tego samego zbioru w każdym powtórzeniu
        n_classes=2,  # liczba klas problemu
    )

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = RandomClassifier()
clf.fit(X_train,y_train)
predict = clf.predict(X_test)
score= accuracy_score(y_test,predict)

print("Accuracy score: \n%.2f" % score)

############################################ZAD 2.2 ####################################################################

class NeighborClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,  parametr = "param"):
        self.parametr = parametr

    def fit(self,X , y ):
        X,y = check_X_y(X,y)
        self.classes_ = unique_labels(y)
        self.X_, self.y_ = X,y

        self.class_ = []
        self.n = X.shape[1]

        return self




    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)

        self.y_pred = []

        for i in range(0, len(X)):
            temporary = []
            for j in range(0, len(self.y_)):
                x = pow((self.X_[j,0]- X[i,0]),2)
                y = pow((self.X_[j,1] - X[i,1]),2)
                x_y = np.sqrt(x+y)
                temporary.append(x_y)
            minim_ = np.min(temporary)
            ix = temporary.index(minim_)
            self.y_pred.append(self.y_[ix])
        return self.y_pred

clf_2 = NeighborClassifier()
clf_2.fit(X_train,y_train)
predict_2 = clf_2.predict(X_test)
score_neighbour = accuracy_score(y_test,predict_2)

clf_3 = KNeighborsClassifier(n_neighbors=1,algorithm='brute')
clf_3.fit(X_train,y_train)
predict_3 = clf_3.predict(X_test)
score_Kneighbour = accuracy_score(y_test, predict_3)

print("Accuracy score own neighbors algorithm: \n%.2f" % score_neighbour)
print("Accuracy score Kneighbors algorithm: \n%.2f" % score_Kneighbour)

############################################ZAD 2.3 ####################################################################


Moons = make_moons()
Circles = make_circles()
Blobs = make_blobs()

n_splits = 2
n_repeats = 5

rskf = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=n_repeats, random_state=1234)


random = RandomClassifier()
neighbour = NeighborClassifier()

temp = [random, neighbour]
data_sets = [Moons,Circles,Blobs]
data_name = ["Moons","Circles","Blobs" ]
scores = []


tabel_mean = []
tabel_std = []

for g in range(0,2):
    for o in range (0,3):
        tmp = data_sets[o]
        X = tmp[0]
        y = tmp[1]
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clf = temp[g]
            clf.fit(X_train, y_train)
            predict = clf.predict(X_test)
            scores.append(accuracy_score(y_test, predict))
            mean_score = np.mean(scores)

            std_score = np.std(scores)
        tabel_std.append(round(std_score,3))
        tabel_mean.append(round(mean_score, 3))
        # print(f"Accuracy score {data_name[o]} {temp[g]}: %.3f,  (%.3f)" % (mean_score, std_score))

print("Average Accuracy:")
p = PrettyTable()
p.field_names = ["Pool name","RandomClassifier","NeighborClassifier"]
for i in range(0,3):
    kk=i+3
    p.add_row([data_name[i],tabel_mean[i],tabel_mean[kk]])
print(p,"\n")

print("Standard Deviation:")
pp = PrettyTable()
pp.field_names = ["Pool name","RandomClassifier","NeighborClassifier"]
for i in range(0,3):
    kk=i+3
    pp.add_row([data_name[i],tabel_std[i],tabel_std[kk]])
print(pp)

