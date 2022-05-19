import numpy as np
from nauka_5_1 import RandomSubspaceEnsembleOwn
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.base import clone
from scipy.stats import rankdata
from scipy.stats import ttest_rel
from tabulate import tabulate
from scipy.stats import ranksums
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from nauka_5_3 import RandomSubspaceEnsemble


# dataset = 'ionosphere'
# dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
# X = dataset[:, :-1]
# y = dataset[:, -1].astype(int)
#
# print("Całkowita liczba atrybutów:", X.shape[1])
#
# n_splits = 5
# n_repeats = 10
# rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
#
# clf = RandomSubspaceEnsembleOwn(hard_voting = True, scales=False)
# scores = []
# for train, test in rskf.split(X, y):
#     clf.fit(X[train], y[train])
#     y_pred = clf.predict(X[test])
#     scores.append(accuracy_score(y[test], y_pred))
# # for train, test in rskf.split(X,y):
# #     clf.fit(X[train], y[train])
# #     y_pred = clf.predict(X[test])
# #     scores.append(accuracy_score(y[test], y_pred))
#
# print("Bagging Wlasny: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
#
# clf1 = DecisionTreeClassifier(random_state=1234)
# scores_CART = []
#
# for train, test in rskf.split(X,y):
#     clf1.fit(X[train], y[train])
#     y_pred = clf1.predict(X[test])
#     scores_CART.append(accuracy_score(y[test],y_pred))
# print("CART: %.3f (%.3f)" % (np.mean(scores_CART), np.std(scores_CART)))
#
# clf2 = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=1234),random_state=1234)
# scores_Bagging = []
#
# for train, test in rskf.split(X,y):
#     clf2.fit(X[train], y[train])
#     y_pred = clf2.predict(X[test])
#     scores_Bagging.append(accuracy_score(y[test],y_pred))
# print("Bagging gotowy: %.3f (%.3f)" % (np.mean(scores_Bagging), np.std(scores_Bagging)))
#
# ######################### ZAD 2 ###########################
#
# clfs = {
#     'scales=F; hard_voting=F': RandomSubspaceEnsembleOwn(scales=False, hard_voting=False),
#     'scales=T; hard_voting=F': RandomSubspaceEnsembleOwn(scales=True, hard_voting=False),
#     'scales=T; hard_voting=T': RandomSubspaceEnsembleOwn(scales=True, hard_voting=True),
#     'scales=F; hard_voting=T': RandomSubspaceEnsembleOwn(scales=False, hard_voting=True),
# }
#
# datasets = ['australian', 'balance']
#
# n_datasets = len(datasets)
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=4413)
#
# scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))
#
# for data_id, dataset in enumerate(datasets):
#    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
#    X = dataset[:, :-1]
#    y = dataset[:, -1].astype(int)
#
#    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#        for clf_id, clf_name in enumerate(clfs):
#            clf = clone(clfs[clf_name])
#            clf.fit(X[train], y[train])
#            y_pred = clf.predict(X[test])
#            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)
#
# print("\nScores:\n", scores.shape)
# np.save('results', scores)
#
# mean_scores = np.mean(scores, axis=2).T
# ranks = []
# for ms in mean_scores:
#     ranks.append(rankdata(ms).tolist())
# ranks = np.array(ranks)
# mean_ranks = np.mean(ranks, axis=0)
#
# alfa = .05
# t_statistic = np.zeros((len(clfs), len(clfs)))
# p_value = np.zeros((len(clfs), len(clfs)))
#
# for i in range(len(clfs)):
#     for j in range(len(clfs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(ranks.T[i], ranks.T[j])
#
# headers = list(clfs.keys())
# names_column = np.expand_dims(np.array(list(clfs.keys())), axis=1)
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("\nt-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
#
# advantage = np.zeros((len(clfs), len(clfs)))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("\nAdvantage:\n", advantage_table)
#
# significance = np.zeros((len(clfs), len(clfs)))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("\nStatistical significance (alpha = 0.05):\n", significance_table)
#
# mean_scores = np.mean(scores, axis=2).T
# print("\nMean scores:\n", mean_scores)

###################### ZAD 3 ##########################################

dataset = 'ionosphere'
dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
X = dataset[:, :-1]
y = dataset[:, -1].astype(int)



n_datasets = len(dataset)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=True, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Hard voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

clf = RandomSubspaceEnsemble(base_estimator=GaussianNB(), n_estimators=10, n_subspace_features=5, hard_voting=False, random_state=123)
scores = []
for train, test in rskf.split(X, y):
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])
    scores.append(accuracy_score(y[test], y_pred))
print("Soft voting - accuracy score: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))