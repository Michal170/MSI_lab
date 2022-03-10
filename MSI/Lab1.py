from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


X, y = datasets.make_classification(
    n_samples=400,  # liczba generowanych wzorców
    n_features=2,  # liczba atrybutów zbioru
    n_informative=2,  # liczba atrybutów informatywnych, tych które zawierają informacje przydatne dla klasyfikacji
    n_repeated=0,  # liczba atrybutów powtórzonych, czyli zduplikowanych kolumn
    n_redundant=0,  # liczba atrybutów nadmiarowych
    flip_y=.08,  # poziom szumu
    random_state=1410,  # ziarno losowości, pozwala na wygenerowanie dokładnie tego samego zbioru w każdym powtórzeniu
    # n_clusters_per_class=1,  # liczba centroidów, a więc l.skupisk każdej z klas problemu
    n_classes=2  # liczba klas problemu

)
plt.figure(figsize=(5, 2.5))
plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
plt.xlabel("$x^1$")
plt.ylabel("$x^2$")
plt.tight_layout()
plt.savefig('zad1.jpg')
plt.show()
plt.savefig("zad_1.png")

#dodanie kolumny z etykietami
datasets = np.concatenate((X, y[:, np.newaxis]), axis=1)

#zapis do pliku csv
np.savetxt(
    "dataset.csv",
    datasets,
    delimiter=",",
    fmt=["%.5f" for i in range(X.shape[1])] + ["%i"]
)

print(datasets)
# print(X.shape, y.shape)

######################################################################################################## 1.2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1420)

clf = GaussianNB()
clf.fit(X_train, y_train)

#wyznaczenie macierzy wsparcia
cl_probabilities = clf.predict_proba(X_test)

predict = np.argmax(cl_probabilities, axis=1)
print("Predykcja klasyfikatora:\n", predict)



score = accuracy_score(y_test, predict)
print("Accuracy score:\n%.2f" % score)

fig, ax = plt.subplots(1,2,figsize=(5,5))

ax[0].scatter(X_test[:,0],X_test[:,1], c=y_test, cmap="bwr")

ax[1].scatter(X_test[:,0],X_test[:,1], c=predict, cmap="bwr")


plt.tight_layout()
# plt.show()
###############################################################################1.3

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1410)
scores = []

# skf.get_n_splits(X,y)

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)

print("Accuracy score:%.3f (%.3f)" % (mean_score,std_score))