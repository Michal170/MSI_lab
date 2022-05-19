from sklearn.tree import DecisionTreeClassifier
from Bagging import Bagging
from Bagging import RandomSubspaceEnsemble
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from scipy.stats import rankdata
from tabulate import tabulate
from scipy.stats import ranksums

dataset = 'ionosphere'
dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)
n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2137)







#----------zadanie 1------------

clf=Bagging(hard_voting=True,wagi=False)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Bagging własny(false): %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf=Bagging(hard_voting=True,wagi=False)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Bagging własny(false): %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
clf=DecisionTreeClassifier(random_state=2137)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))

print("CART: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=2137),random_state=2137)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Bagging gotowy: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
"""
#--------------zadanie 2-------------------
#porownanie zespolu z kombinacjami flag znajdujacymi sie w team
"""
team = {
    'bez wag bez wektorow': Bagging(wagi=False,hard_voting=False),
    'z wagami bez wektorow': Bagging(wagi=True,hard_voting=False),
    'z wagami z wektorami': Bagging(wagi=True,hard_voting=True),
    'bez wag z wektorami': Bagging(wagi=False,hard_voting=True),
}

datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'glass2', 'heart', 'ionosphere',
            'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean',
            'vowel0', 'waveform', 'wisconsin', 'yeast3']

n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=2137)

scores = np.zeros((len(team), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(team):
            clf = clone(team[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

print("\nScores:\n", scores.shape)
np.save('results', scores)


scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)
mean_ranks = np.mean(ranks, axis=0)

alfa = .05
w_statistic = np.zeros((len(team), len(team)))
p_value = np.zeros((len(team), len(team)))

for i in range(len(team)):
    for j in range(len(team)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])


headers = list(team.keys())
names_column = np.expand_dims(np.array(list(team.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)


advantage = np.zeros((len(team), len(team)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(team), len(team)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)
"""
#--------zadanie 3--------------- implementacja znajduje sie w Bagging
"""
clf=RandomSubspaceEnsemble(base_estimator=DecisionTreeClassifier(random_state=2137),random_state=2137,hard_voting=False)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Zespół dokonujący dywersyfikacji modeli bazowych: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
