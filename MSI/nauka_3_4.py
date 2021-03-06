import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest



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

X = SelectKBest(k=6).fit_transform(X,y)


scores = np.zeros((len(clfs), n_splits * n_repeats))

for fold_id, (train, test) in enumerate(rskf.split(X, y)):
    for clf_id, clf_name in enumerate(clfs):
        clfs[clf_name].fit(X[train], y[train])
        y_pred = clfs[clf_name].predict(X[test])
        scores[clf_id, fold_id] = accuracy_score(y[test], y_pred)
mean = np.mean(scores, axis=1)
std = np.std(scores, axis=1)

for clf_id, clf_name in enumerate(clfs):
    print("%s  po zastosowaniu redukcji : %.3f (%.2f)" % (clf_name, round(mean[clf_id],2), std[clf_id]))
print(f"X shape po redukcji: {X.shape}")