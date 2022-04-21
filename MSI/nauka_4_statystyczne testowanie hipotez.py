import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

X,y = datasets.make_classification(
    n_samples=1000,
    n_classes=10,
    n_informative=10,

)

# vector = np.random.normal(size=20)
# X = X * vector



clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'SVC': SVC(),
    'CART' : DecisionTreeClassifier(random_state=1111),
}

datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'iris', 'heart', 'ionosphere',
            'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean',
            'spambase', 'waveform', 'wine', 'yeast6']

n_datasets = len(datasets)

n_splits = 2
n_repeats = 5

rskf = RepeatedStratifiedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=1111)
scores = np.zeros((len(clfs),n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            # clfs[clf_name].fit(X[train], y[train])
            # y_pred = clfs[clf_name].predict(X[test])
            # scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
            scores[clf_id,data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)
    # mean = np.mean(scores, axis=1)
    # std = np.std(scores, axis=1)
    # print("Zadanie 1:\n")
    # for clf_id, clf_name in enumerate(clfs):
    #     print("%s: %.3f (%.3f)" % (clf_name, mean[clf_id], std[clf_id]))