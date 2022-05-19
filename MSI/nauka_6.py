import numpy as np
from sklearn import clone
from sklearn.datasets import make_classification
from scipy.stats import ttest_rel
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from tabulate import tabulate




X1, y1 = make_classification(
    n_samples=700,
    n_classes=2,
    flip_y=.15,
    weights= [0.17,0.83]
)

X2, y2 = make_classification(
    n_samples=700,
    n_classes=2,
    flip_y=.15,
    weights= [0.1,0.9]
)

X3, y3 = make_classification(
    n_samples=700,
    n_classes=2,
    flip_y=.15,
    weights= [0.01,0.99]
)

X4, y4 = make_classification(
    n_samples=700,
    n_classes=3,
    n_clusters_per_class=1,
    flip_y=.15,
    weights= [0.09,0.45,0.45]
)

X5, y5 = make_classification(
    n_samples=700,
    n_features=20,
    n_informative=2,
    n_classes=2,
    flip_y=.15,
)



clf = DecisionTreeClassifier(random_state=4413)

preprocs = {
    'none': None,
    'ros': RandomOverSampler(random_state=4413),
    'smote' : SMOTE(random_state=4413),
    'rus': RandomUnderSampler(random_state=4413),
}

metrics = {
    "recall": recall,
    'precision': precision,
    'specificity': specificity,
    'f1': f1_score,
    'g-mean': geometric_mean_score_1,
    'bac': balanced_accuracy_score,
}

headers = ["none", "ros", "smote","rus"]
names_column = np.array([["recall"], ["precision"], ["specificity"], ["f1"], ["g-mean"], ["bac"]])

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)

scores1 = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
scores2 = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
scores3 = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
scores4 = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
scores5 = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))

for fold_id, (train, test) in enumerate(rskf.split(X1, y1)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X1[train], y1[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X1[train], y1[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X1[test])

        for metric_id, metric in enumerate(metrics):
            scores1[preproc_id, fold_id, metric_id] = metrics[metric](
                y1[test], y_pred)

for fold_id, (train, test) in enumerate(rskf.split(X2, y2)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X2[train], y2[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X2[train], y2[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X2[test])

        for metric_id, metric in enumerate(metrics):
            scores2[preproc_id, fold_id, metric_id] = metrics[metric](
                y2[test], y_pred)

for fold_id, (train, test) in enumerate(rskf.split(X3, y3)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X3[train], y3[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X3[train], y3[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X3[test])

        for metric_id, metric in enumerate(metrics):
            scores3[preproc_id, fold_id, metric_id] = metrics[metric](
                y3[test], y_pred)

for fold_id, (train, test) in enumerate(rskf.split(X4, y4)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X4[train], y4[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X4[train], y4[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X4[test])

        for metric_id, metric in enumerate(metrics):
            scores4[preproc_id, fold_id, metric_id] = metrics[metric](
                y4[test], y_pred)

for fold_id, (train, test) in enumerate(rskf.split(X5, y5)):
    for preproc_id, preproc in enumerate(preprocs):
        clf = clone(clf)

        if preprocs[preproc] == None:
            X_train, y_train = X5[train], y5[train]
        else:
            X_train, y_train = preprocs[preproc].fit_resample(
                X5[train], y5[train])

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X5[test])

        for metric_id, metric in enumerate(metrics):
            scores5[preproc_id, fold_id, metric_id] = metrics[metric](
                y5[test], y_pred)


np.save('results1', scores1)
np.save('results2', scores2)
np.save('results3', scores3)
np.save('results4', scores4)
np.save('results5', scores5)

#scores1 = np.load("results1.npy")
scores1 = np.mean(scores1, axis=1).T

#scores2 = np.load("results2.npy")
scores2 = np.mean(scores2, axis=1).T

#scores3 = np.load("results3.npy")
scores3 = np.mean(scores3, axis=1).T

#scores4 = np.load("results4.npy")
scores4 = np.mean(scores4, axis=1).T

#scores5 = np.load("results5.npy")
scores5 = np.mean(scores5, axis=1).T

table1 = np.concatenate((names_column, scores1), axis=1)
table1 = tabulate(table1, headers, floatfmt=".2f")
print("zbior 1:\n", table1)

table2 = np.concatenate((names_column, scores2), axis=1)
table2 = tabulate(table2, headers, floatfmt=".2f")
print("zbior 2:\n", table2)

table3 = np.concatenate((names_column, scores3), axis=1)
table3 = tabulate(table3, headers, floatfmt=".2f")
print("zbior 3:\n", table3)

table4 = np.concatenate((names_column, scores4), axis=1)
table4 = tabulate(table4, headers, floatfmt=".2f")
print("zbior 4:\n", table4)

table5 = np.concatenate((names_column, scores5), axis=1)
table5 = tabulate(table5, headers, floatfmt=".2f")
print("zbior 5:\n", table5)

headers_T = ["none", "ros", "smote","rus"]
names_column_T = np.array([["none"], ["ros"], ["smote"],['rus']])

alfa = .05
t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

print("Testy parowe dla zbioru 1:")

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores1[i], scores1[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column_T, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers_T, floatfmt=".2f")
p_value_table = np.concatenate((names_column_T, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers_T, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column_T, advantage), axis=1), headers_T)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column_T, significance), axis=1), headers_T)
print("Statistical significance (alpha = 0.05):\n", significance_table)

print("Testy parowe dla zbioru 2:")
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores2[i], scores2[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column_T, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers_T, floatfmt=".2f")
p_value_table = np.concatenate((names_column_T, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers_T, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column_T, advantage), axis=1), headers_T)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column_T, significance), axis=1), headers_T)
print("Statistical significance (alpha = 0.05):\n", significance_table)

print("Testy parowe dla zbioru 3:")
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores3[i], scores3[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column_T, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers_T, floatfmt=".2f")
p_value_table = np.concatenate((names_column_T, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers_T, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column_T, advantage), axis=1), headers_T)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column_T, significance), axis=1), headers_T)
print("Statistical significance (alpha = 0.05):\n", significance_table)

print("Testy parowe dla zbioru 4:")
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores4[i], scores4[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column_T, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers_T, floatfmt=".2f")
p_value_table = np.concatenate((names_column_T, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers_T, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column_T, advantage), axis=1), headers_T)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column_T, significance), axis=1), headers_T)
print("Statistical significance (alpha = 0.05):\n", significance_table)

print("Testy parowe dla zbioru 5:")
for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(scores5[i], scores5[j])
print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

t_statistic_table = np.concatenate((names_column_T, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers_T, floatfmt=".2f")
p_value_table = np.concatenate((names_column_T, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers_T, floatfmt=".2f")
print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column_T, advantage), axis=1), headers_T)
print("Advantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column_T, significance), axis=1), headers_T)
print("Statistical significance (alpha = 0.05):\n", significance_table)

#mean_scores = np.mean(scores, axis=2).T
#print("\nMean scores:\n", mean_scores)





































# from sklearn import datasets
#
# from sklearn.tree import DecisionTreeClassifier
# from imblearn.over_sampling import RandomOverSampler, SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from strlearn.metrics import recall, precision, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
# from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn.base import clone
# import numpy as np
# from tabulate import tabulate
# from scipy.stats import ttest_rel
#
# clf = DecisionTreeClassifier(random_state=2137)
#
# preprocs = {
#     'none': None,
#     'ros': RandomOverSampler(random_state=2137),
#     'smote' : SMOTE(random_state=2137),
#     'rus': RandomUnderSampler(random_state=2137),
# }
# metrics = {
#     "recall": recall,
#     'precision': precision,
#     'specificity': specificity,
#     'f1': f1_score,
#     'g-mean': geometric_mean_score_1,
#     'bac': balanced_accuracy_score,
# }
#
# headers_1 = ["none", "ros", "smote","rus"]
# n_column = np.array([["recall"], ["precision"], ["specificity"], ["f1"], ["g-mean"], ["bac"]])
#
# headers = ["none", "ros", "smote","rus"]
# names_column = np.array([["none"], ["ros"], ["smote"],['rus']])
# #1
# print("Dla pierwszego zbioru")
# X, y = datasets.make_classification(
#     n_samples=100,
#     n_features=4,
#     n_classes=2,
#     weights=[0.17,0.83],
#     random_state=2137,
#     n_informative=4,
#     n_redundant=0,
# )
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
#
# scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
#
# for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#     for preproc_id, preproc in enumerate(preprocs):
#         clf = clone(clf)
#
#         if preprocs[preproc] == None:
#             X_train, y_train = X[train], y[train]
#         else:
#             X_train, y_train = preprocs[preproc].fit_resample(
#                 X[train], y[train])
#
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X[test])
#
#         for metric_id, metric in enumerate(metrics):
#             scores[preproc_id, fold_id, metric_id] = metrics[metric](
#                 y[test], y_pred)
#
#
#
# scores = np.mean(scores, axis=1).T
#
#
# table = np.concatenate((n_column, scores), axis=1)
# table = tabulate(table, headers_1, floatfmt=".2f")
# print("dla zbioru 1:\n", table)
#
# alfa = .05
# t_statistic = np.zeros((len(preprocs), len(preprocs)))
# p_value = np.zeros((len(preprocs), len(preprocs)))
#
#
# for i in range(len(preprocs)):
#     for j in range(len(preprocs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
#
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
#
# advantage = np.zeros((len(preprocs), len(preprocs)))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)
#
# significance = np.zeros((len(preprocs), len(preprocs)))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)
#
# # #2
# print("Dla drugiego zbioru")
# X, y= datasets.make_classification(
#     n_samples=100,
#     n_features=6,
#     n_classes=2,
#     weights=[0.1,0.9],
#     random_state=2137,
#     flip_y=.05
# )
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
#
# scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
#
# for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#     for preproc_id, preproc in enumerate(preprocs):
#         clf = clone(clf)
#
#         if preprocs[preproc] == None:
#             X_train, y_train = X[train], y[train]
#         else:
#             X_train, y_train = preprocs[preproc].fit_resample(
#                 X[train], y[train])
#
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X[test])
#
#         for metric_id, metric in enumerate(metrics):
#             scores[preproc_id, fold_id, metric_id] = metrics[metric](
#                 y[test], y_pred)
#
#
# scores = np.mean(scores, axis=1).T

#
# table = np.concatenate((n_column, scores), axis=1)
# table = tabulate(table, headers_1, floatfmt=".2f")
# print("dla zbioru 2:\n", table)
#
# t_statistic = np.zeros((len(preprocs), len(preprocs)))
# p_value = np.zeros((len(preprocs), len(preprocs)))
#
#
# for i in range(len(preprocs)):
#     for j in range(len(preprocs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
#
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
#
# advantage = np.zeros((len(preprocs), len(preprocs)))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)
#
# significance = np.zeros((len(preprocs), len(preprocs)))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)
# #3
# print("Dla trzeciego zbioru")
#
# X, y = datasets.make_classification(
#     n_samples=1000,
#     n_features=6,
#     n_classes=2,
#     weights=[0.01,0.99],
#     random_state=2137,
# )
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
#
# scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
#
# for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#     for preproc_id, preproc in enumerate(preprocs):
#         clf = clone(clf)
#
#         if preprocs[preproc] == None:
#             X_train, y_train = X[train], y[train]
#         else:
#             X_train, y_train = preprocs[preproc].fit_resample(
#                 X[train], y[train])
#
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X[test])
#
#         for metric_id, metric in enumerate(metrics):
#             scores[preproc_id, fold_id, metric_id] = metrics[metric](
#                 y[test], y_pred)
#
#
# scores = np.mean(scores, axis=1).T
#
#
# table = np.concatenate((n_column, scores), axis=1)
# table = tabulate(table, headers_1, floatfmt=".2f")
# print("dla zbioru 3:\n", table)
#
# t_statistic = np.zeros((len(preprocs), len(preprocs)))
# p_value = np.zeros((len(preprocs), len(preprocs)))
#
#
# for i in range(len(preprocs)):
#     for j in range(len(preprocs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
#
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
#
# advantage = np.zeros((len(preprocs), len(preprocs)))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)
#
# significance = np.zeros((len(preprocs), len(preprocs)))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)
# #4
# print("Dla czwartego zbioru")
# X, y = datasets.make_classification(
#     n_samples=1000,
#     n_features=6,
#     n_clusters_per_class=1,
#     n_classes=3,
#     weights=[0.09,0.45,0.45],
#     random_state=2137,
# )
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(
#     n_splits=n_splits, n_repeats=n_repeats, random_state=42)
#
# scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
#
# for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#     for preproc_id, preproc in enumerate(preprocs):
#         clf = clone(clf)
#
#         if preprocs[preproc] == None:
#             X_train, y_train = X[train], y[train]
#         else:
#             X_train, y_train = preprocs[preproc].fit_resample(
#                 X[train], y[train])
#
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X[test])
#
#         for metric_id, metric in enumerate(metrics):
#             scores[preproc_id, fold_id, metric_id] = metrics[metric](
#                 y[test], y_pred)
#
#
# scores = np.mean(scores, axis=1).T
#
#
# table = np.concatenate((n_column, scores), axis=1)
# table = tabulate(table, headers_1, floatfmt=".2f")
# print("dla zbioru 4:\n", table)
#
# t_statistic = np.zeros((len(preprocs), len(preprocs)))
# p_value = np.zeros((len(preprocs), len(preprocs)))
#
#
# for i in range(len(preprocs)):
#     for j in range(len(preprocs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
#
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
#
# advantage = np.zeros((len(preprocs), len(preprocs)))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)
#
# significance = np.zeros((len(preprocs), len(preprocs)))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)
# #5
# print("Dla piatego zbioru")
# X, y = datasets.make_classification(
#     n_samples=100,
#     n_features=20,
#     n_informative=2,
#     n_classes=2,
#     random_state=2137,
# )
# n_splits = 5
# n_repeats = 2
# rskf = RepeatedStratifiedKFold(
#     n_splits=n_splits, n_repeats=n_repeats, random_state=42)
#
# scores = np.zeros((len(preprocs), n_splits * n_repeats, len(metrics)))
#
# for fold_id, (train, test) in enumerate(rskf.split(X, y)):
#     for preproc_id, preproc in enumerate(preprocs):
#         clf = clone(clf)
#
#         if preprocs[preproc] == None:
#             X_train, y_train = X[train], y[train]
#         else:
#             X_train, y_train = preprocs[preproc].fit_resample(
#                 X[train], y[train])
#
#         clf.fit(X_train, y_train)
#         y_pred = clf.predict(X[test])
#
#         for metric_id, metric in enumerate(metrics):
#             scores[preproc_id, fold_id, metric_id] = metrics[metric](
#                 y[test], y_pred)
#
#
# scores = np.mean(scores, axis=1).T
#
#
# table = np.concatenate((n_column, scores), axis=1)
# table = tabulate(table, headers_1, floatfmt=".2f")
# print("dla zbioru 5:\n", table)
#
# t_statistic = np.zeros((len(preprocs), len(preprocs)))
# p_value = np.zeros((len(preprocs), len(preprocs)))
#
#
# for i in range(len(preprocs)):
#     for j in range(len(preprocs)):
#         t_statistic[i, j], p_value[i, j] = ttest_rel(scores[i], scores[j])
# print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)
#
# t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
# t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
# p_value_table = np.concatenate((names_column, p_value), axis=1)
# p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
# print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)
#
# advantage = np.zeros((len(preprocs), len(preprocs)))
# advantage[t_statistic > 0] = 1
# advantage_table = tabulate(np.concatenate(
#     (names_column, advantage), axis=1), headers)
# print("Advantage:\n", advantage_table)
#
# significance = np.zeros((len(preprocs), len(preprocs)))
# significance[p_value <= alfa] = 1
# significance_table = tabulate(np.concatenate(
#     (names_column, significance), axis=1), headers)
# print("Statistical significance (alpha = 0.05):\n", significance_table)