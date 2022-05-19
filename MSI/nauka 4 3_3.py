import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import ttest_rel
from tabulate import tabulate
from scipy.stats import rankdata
from scipy.stats import ranksums


datasets = ['australian', 'balance', 'breastcan', 'cryotherapy', 'diabetes',
            'digit', 'ecoli4', 'german', 'iris', 'heart', 'ionosphere',
            'liver', 'monkthree', 'shuttle-c0-vs-c4', 'sonar', 'soybean',
            'spambase', 'waveform', 'wine', 'yeast6']
clfs = {
    'GNB': GaussianNB(),
    'kNN': KNeighborsClassifier(),
    'SVC': SVC(),
    'CART' : DecisionTreeClassifier(random_state=1111),
}

scores = np.load('results.npy')


# for data_id, dataset in enumerate(datasets):
#     table_1_3 = scores[:, data_id, :]


alfa = .05
t_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

scores = np.load('results.npy')
print("\nScores:\n", scores.shape)

# for data_id, dataset in enumerate(datasets):
#     table_ = scores[:, data_id, :]

# mean_scores = np.mean(scores, axis=2).T
mean_scores = np.mean(scores, axis=2)

print(mean_scores.shape)


for i in range(len(clfs)):
    for j in range(len(clfs)):
        # t_statistic[i, j], p_value[i, j] = ttest_rel(table_[i], table_[j])
        t_statistic[i, j], p_value[i, j] = ttest_rel(mean_scores[i], mean_scores[j])


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


# mean_scores = np.mean(scores, axis=2).T

ranks = []
for i in mean_scores:
    ranks.append((rankdata(i).tolist()))
ranks = np.array(ranks)

w_statistic = np.zeros((len(clfs), len(clfs)))
p_value = np.zeros((len(clfs), len(clfs)))

for i in range(len(clfs)):
    for j in range(len(clfs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])


w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statystyka:\n", w_statistic_table)