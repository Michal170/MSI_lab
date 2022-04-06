import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



X,y = datasets.make_classification(
    n_samples=500,


)
vector = np.random.normal(size=(20))

for i in range(0,500):
    for j in range(len(vector)):
        X[i,j]=X[i,j]*vector[j]




clfs = {
    'GNB' : GaussianNB(),
    'kNN' : KNeighborsClassifier(),
    'CART' : DecisionTreeClassifier(random_state=1234)
}

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=42)
scores = np.zeros((len(clfs), n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):


        clfs[clf_name].fit(X[train], y[train])
        y_pred = clfs[clf_name].predict(X[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

#########################ZADANIE 3.2####################################

scores_2 = np.zeros((len(clfs), n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        scaler = StandardScaler()
        scaler.fit(X[train])
        X_train = scaler.transform(X[train])
        X_test = scaler.transform(X[test])
        clfs[clf_name].fit(X_train, y[train])
        y_pred = clfs[clf_name].predict(X_test)
        scores_2[clf_id, fold_id] = accuracy_score(y[test], y_pred)
mean = np.mean(scores_2, axis=1)
std = np.std(scores_2, axis=1)

print("==================================\nZadanie nr 2:")
for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))

############################# ZADANIE 3.3 ##################################33

pca = PCA(svd_solver = 'full')
pca.fit(X)
accumulated_sum = np.cumsum(pca.explained_variance_ratio_)
attributed_mask = accumulated_sum < .8
X_pca = X[:,attributed_mask]

scores_3 = np.zeros((len(clfs), n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X_pca, y)):
    for clf_id, clf_name in enumerate(clfs):
        scaler = StandardScaler()
        scaler.fit(X_pca[train])
        X_train = scaler.transform(X_pca[train])
        X_test = scaler.transform(X_pca[test])
        clfs[clf_name].fit(X_train, y[train])
        y_pred = clfs[clf_name].predict(X_test)
        scores_3[clf_id, fold_id] = accuracy_score(y[test], y_pred)
mean = np.mean(scores_3, axis=1)
std = np.std(scores_3, axis=1)


print("==================================\nZadanie nr 3:")
for clf_id, clf_name in enumerate(clfs):
    print("%s: %.3f (%.2f)" % (clf_name, mean[clf_id], std[clf_id]))