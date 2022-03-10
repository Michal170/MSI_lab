from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

print("############### Zad.1.1 #######################")

X, y = datasets.make_classification(
    n_features=2,  # liczba atrybutów zbioru
    n_samples=400,  # liczba generowanych wzorców
    n_informative=2,  # liczba atrybutów informatywnych, tych które zawierają informacje przydatne dla klasyfikacji
    n_repeated=0,  # liczba atrybutów powtórzonych, czyli zduplikowanych kolumn
    n_redundant=0,  # liczba atrybutów nadmiarowych
    flip_y=.08,  # poziom szumu
    random_state=1410,  # ziarno losowości, pozwala na wygenerowanie dokładnie tego samego zbioru w każdym powtórzeniu
    n_classes=2,  # liczba klas problemu

)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr")
ax.set_xlabel("feature 0")
ax.set_ylabel("feature 1")
ax.set_title("Dataset_Zad.1.1")
plt.tight_layout()
# plt.show()
plt.savefig("zad_1.png")

datasets = np.concatenate((X, y[:, np.newaxis]), axis=1)  # dodanie kolumny z etykietami

np.savetxt(  # zapis do pliku csv
    "dataset.csv",
    datasets,
    delimiter=",",
    fmt=["%.5f" for i in range(X.shape[1])] + ["%i"]
)

print("Datasets:\n", datasets, "\n")
print("X.shape, y.shape:", X.shape, y.shape)
plt.show()

########################################## ZADANIE 1.2 ##############################################################

print("############### Zad.1.2 #######################")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)
clf = GaussianNB()  # gaussowski naiwny klasyfikator
clf.fit(X_train, y_train)

cl_probabilities = clf.predict_proba(X_test)  # wyznaczenie macierzy wsparcia
predict = np.argmax(cl_probabilities, axis=1)  # predykcja klasyfikatora dla zbioru testowego
print("Predykcja klasyfikatora:\n", predict)
print("w. rzeczywiste:\n", y_test, "\n")

score = accuracy_score(y_test, predict)
print("Accuracy score:\n%.2f" % score)

fig, ax = plt.subplots(1, 2, figsize=(5, 5))

ax[0].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="bwr")
ax[0].set_xlabel("Feature 0")
ax[0].set_ylabel("Feature 1")
ax[0].set_title("Etykiety rzeczywiste ")
ax[0].set_xlim(-4,4)
ax[0].set_ylim(-4,4)

ax[1].scatter(X_test[:, 0], X_test[:, 1], c=predict, cmap="bwr")
ax[1].set_xlabel("Feature 0")
ax[1].set_ylabel("Feature 1")
ax[1].set_title("Etykiety predykcji")
ax[1].set_xlim(-4,4)
ax[1].set_ylim(-4,4)

plt.tight_layout()
plt.show()
plt.savefig("zad_2.png")

######################################### ZADANIE 1.3 #################################################

print("############### Zad.1.3 #######################")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)
scores = []

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    scores.append(accuracy_score(y_test, predict))

mean_score = np.mean(scores)
std_score = np.std(scores)

print("Accuracy score: %.3f,  (%.3f)" % (mean_score, std_score))
