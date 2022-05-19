# from Lab5_1 import OwnBaggingClasifier
from Lab5_2 import OwnBaggingClasifier
from Lab5_3 import RandomSubspaceEnsemble
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import rankdata
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
# from CARTEnsemble import ownCARTEnsemble
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_rel
from sklearn.base import clone

# #
# # clf1 = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=False,weights=True),
# # clf2 = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=False,weights=False),
# # clf3 = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=True,weights=False),
# # clf4 = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=True,weights=True),
#
# datasets = ['appendicitis', 'balance', 'banana', 'bupa']
#
# clfs = {
#     'clf_1': OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=False,weights=True),
#     'clf_2': OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=False,weights=False),
#     'clf_3' : OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=True,weights=False),
#     'clf_4' : OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=True,weights=True),
# }
#
# n_datasets = len(datasets)
#
# n_splits = 2
# n_repeats = 5
#
# rskf = RepeatedStratifiedKFold(
#     n_splits=n_splits,
#     n_repeats=n_repeats,
#     random_state=1111)
# scores = np.zeros((n_datasets,n_splits * n_repeats,(len(clfs))))
#
# for data_id, dataset in enumerate(datasets):
#     dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
#     X = dataset[:, :-1]
#     y = dataset[:, -1].astype(int)
#
#     for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#         for clf_id, clf_name in enumerate(clfs):
#             clf = clone(clfs[clf_name])
#             clf.fit(X[train], y[train])
#             y_pred = clf.predict(X[test])
#             scores[data_id,fold_id,clf_id] = accuracy_score(y[test], y_pred)
#
# alfa = .05
# t_statistic = np.zeros((len(clfs), len(clfs)))
# p_value = np.zeros((len(clfs), len(clfs)))
#
# scores = np.load('results.npy')
# print("\nScores:\n", scores.shape)
# table_1=scores[1,:,:]
# # mean = np.mean(table_1, axis=1)
# # std = np.std(table_1, axis=1)
# # print("mean:\n",mean,"\n",std)
#
#
#
# for i in range(len(clfs)):
#     for j in range(len(clfs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(table_1[i], table_1[j])
#
# # print("t-statystyka:\n", t_statistic, "\np-wartosc:\n", p_value)
#
#
#
# headers = ["clf_1", "clf_2", "clf_3","clf_4"]
# names_column = np.array([["clf_1"], ["clf_2"], ["clf_3"],["clf_4"]])
#
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("Tabela t-statystyki:\n", t_statistic_table, "\n\nTabela p-wartości:\n",p_value_table)
#



datasets = ['ring','iris','led7digit','texture']
# print(datasets[0])
#
# for data_id, dataset in enumerate(datasets):
#     dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
#     X = dataset[:, :-1]
#     y = dataset[:, -1].astype(int)
#
# # datasets = ['australian']
# #
# # datasets = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=",")
# # X = datasets[:, :-1]
# # y = datasets[:, -1].astype(int)
#
n_datasets = len(datasets)
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(
#     n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores_1 = []
mean_1 = []
std_1 = []
scores_2 = []
mean_2 = []
std_2 = []
scores_3 = []
mean_3 = []
std_3 = []
scores_4 = []
mean_4 = []
std_4 = []
scores = []
mean = []
std = []
scores_5 = []
mean_5 = []
std_5 = []
t_statistic = np.zeros((len(datasets), len(datasets)))
p_value = np.zeros((len(datasets), len(datasets)))

for k in range(0,n_datasets):


    dataset = np.genfromtxt("datasets/%s.csv" % (datasets[k]), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    # datasets = ['australian']
    #
    # datasets = np.genfromtxt("datasets/%s.csv" % (datasets), delimiter=",")
    # X = datasets[:, :-1]
    # y = datasets[:, -1].astype(int)

    n_datasets = len(datasets)
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    print("Dataset:",datasets[k])

    clf = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=False,weights=True)
    # scores = []
    # mean = []

    for train, test in rskf.split(X, y):
        clf.fit(X[train],y[train])
        y_pred = clf.predict(X[test])
        scores.append(accuracy_score(y[test], y_pred))
    mean.append(round(np.mean(scores),4))
    std.append(round(np.std(scores), 3))

    print("hard_voting=False,weights=True : %.3f (%.3f)" % (np.mean(scores), np.std(scores)))

    clf1 = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123,hard_voting=False,weights=False)
    # scores_1 = []
    # mean_1 = []
    for train, test in rskf.split(X, y):
        clf1.fit(X[train],y[train])
        y_pred = clf1.predict(X[test])
        scores_1.append(accuracy_score(y[test], y_pred))
    mean_1.append(round(np.mean(scores_1), 4))
    std_1.append(round(np.std(scores_1), 3))
    print("hard_voting=False,weights=False : %.3f (%.3f)" % (np.mean(scores_1), np.std(scores_1)))

    clf3 = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123, hard_voting=True,
                              weights=False)
    # scores_3 = []
    # mean_3 = []
    for train, test in rskf.split(X, y):
        clf3.fit(X[train], y[train])
        y_pred = clf3.predict(X[test])
        scores_3.append(accuracy_score(y[test], y_pred))
    mean_3.append(round(np.mean(scores_3), 4))
    std_3.append(round(np.std(scores_3), 3))
    print("hard_voting=True, weights=False : %.3f (%.3f)" % (np.mean(scores_3), np.std(scores_3)))

    clf4 = OwnBaggingClasifier(base_estimator=GaussianNB(), n_estimators=5, random_state=123, hard_voting=True,
                               weights=True)
    # scores_4 = []
    # mean_4 = []
    for train, test in rskf.split(X, y):
        clf4.fit(X[train], y[train])
        y_pred = clf4.predict(X[test])
        scores_4.append(accuracy_score(y[test], y_pred))
    mean_4.append(round(np.mean(scores_4), 4))
    std_4.append(round(np.std(scores_4), 3))

    print("hard_voting=True, weights=True : %.3f (%.3f)" % (np.mean(scores_4), np.std(scores_4)))

    clf2 = DecisionTreeClassifier()
    # scores_2 = []
    # mean_2 = []
    for train, test in rskf.split(X, y):
        clf2.fit(X[train],y[train])
        y_pred = clf2.predict(X[test])
        scores_2.append(accuracy_score(y[test], y_pred))
    mean_2.append(round(np.mean(scores_2), 4))
    std_2.append(round(np.std(scores_2), 3))


    print("Pojedyńcze drzewo w datasecie: %.3f (%.3f)" % (np.mean(scores_2), np.std(scores_2)))

    clf5 = RandomSubspaceEnsemble(base_estimator=GaussianNB(),random_state=123)
    # scores_5 = []
    # mean_5 = []
    for train, test in rskf.split(X, y):
        clf5.fit(X[train], y[train])
        y_pred = clf5.predict(X[test])
        scores_5.append(accuracy_score(y[test], y_pred))
    mean_5.append(round(np.mean(scores_5), 4))
    std_5.append(round(np.std(scores_5), 3))

    print("Zad 3 : %.3f (%.3f)" % (np.mean(scores_5), np.std(scores_5)))


# headers_T = ['ring','iris','led7digit','texture']
# names_column_T = np.array([["ring"], ["iris"], ["led7digit"],['texture']])
#
# for i in range(4):
#     for j in range(4):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
# t_statistic_table = np.concatenate((names_column_T, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers_T, floatfmt=".2f")
# p_value_table = np.concatenate((names_column_T, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers_T, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)


# print(mean)
print("\n\nTabela średniej")
table = PrettyTable()
table.field_names = ["dataset","h_v=False,w=True" ,"h_v=False,w=False","h_v=True, w=False","h_v=True, w=True","dywersyfikacja","pojedyncze drzewo"]
for i in range(0,n_datasets):
    table.add_row([datasets[i],mean[i],mean_1[i],mean_3[i],mean_4[i],mean_5[i],mean_2[i]])

print(table,"\n\n")
print("\n\nTabela odchylenia standardowego")
table_std = PrettyTable()
table_std.field_names = ["dataset","h_v=False,w=True" ,"h_v=False,w=False","h_v=True, w=False","h_v=True, w=True","dywersyfikacja","pojedyncze drzewo"]
for i in range(0,n_datasets):
    table_std.add_row([datasets[i],std[i],std_1[i],std_3[i],std_4[i],std_5[i],std_2[i]])

print(table_std)