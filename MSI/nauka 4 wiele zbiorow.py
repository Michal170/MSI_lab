from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata, ranksums
from tabulate import tabulate




clfs = {
    'GNB' : GaussianNB(),
    'SVM' : SVC(),
    'kNN' : KNeighborsClassifier(),
    'CART' : DecisionTreeClassifier(random_state=1111),
}

datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit']



n_datasets = len(datasets)
n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(clfs), n_datasets, n_splits * n_repeats))

for data_id, dataset in enumerate(datasets):
    dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)

    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for clf_id, clf_name in enumerate(clfs):
            clf = clone(clfs[clf_name])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])
            scores[clf_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

np.save('results', scores)

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

mean_scores = np.mean(scores, axis=2).T
print("\nMean scores:\n", mean_scores)

ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)

print("\nRanks:\n", ranks)

mean_ranks = np.mean(ranks,axis=0)

print("\nMean ranks: \n", mean_ranks)

alfa=.05
w_statistic = np.zeros((len(clfs),len(clfs)))
p_value = np.zeros((len(clfs),len(clfs)))


for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i,j], p_value[i,j] = ranksums(ranks.T[i],ranks.T[j])


headers = list(clfs.keys())
names_column = np.expand_dims(np.array(list(clfs.keys())),axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n",p_value_table)


advantage = np.zeros((len(clfs),len(clfs)))
advantage[w_statistic>0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(clfs),len(clfs)))
significance[p_value <=alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)
