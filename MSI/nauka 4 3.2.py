import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
from scipy.stats import ttest_rel
from tabulate import tabulate


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
#
# datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
#             'digit', 'ecoli4', 'german', 'iris', 'heart', 'ionosphere',
#             'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean',
#             'spambase', 'waveform', 'wine', 'yeast6']
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
# scores = np.zeros((len(clfs),n_datasets, n_splits * n_repeats))
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
#             # clfs[clf_name].fit(X[train], y[train])
#             # y_pred = clfs[clf_name].predict(X[test])
#             # scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
#             scores[clf_id,data_id, fold_id] = accuracy_score(y[test], y_pred)
#
# np.save('results', scores)

alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)
table_1=scores[:,1,:]

for i in range(len(clfs)):
    for j in range(len(clfs)):
        t_statistic[i, j], p_value[i, j] = ttest_rel(table_1[i], table_1[j])

print("t-statystyka:\n", t_statistic, "\np-wartosc:\n", p_value)



headers = ["GNB", "kNN", 'SVC',"CART",]
names_column = np.array([["GNB"], ["kNN"],['SVC'], ["CART"],])

t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("Tabela t-statystyki:\n", t_statistic_table, "\n\nTabela p-wartosci:\n", p_value_table)

advantage = np.zeros((len(clfs), len(clfs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("Tabela przewagi:\n", advantage_table)

significance = np.zeros((len(clfs), len(clfs)))
significance[t_statistic <= alfa] = 1
significance_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("Tabela różnic statystycznych:\n", significance_table)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate((names_column, advantage), axis=1), headers)
print("Tabela końcowych obserwacji:\n", stat_better_table)


