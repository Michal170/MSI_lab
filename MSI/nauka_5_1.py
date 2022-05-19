import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata
import random


class RandomSubspaceEnsembleOwn(BaseEnsemble):

    def __init__(self, base_estimator=DecisionTreeClassifier(random_state=4413),n_subspace_features=5, n_estimators=5, hard_voting=True, random_state=4413, scales=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.scales = scales
        self.n_subspace_features = n_subspace_features
        self.hard_voting = hard_voting
        self.random_state = random_state
        np.random.seed(self.random_state)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_features = X.shape[1]

        if self.scales == True:
            n_splits = 5
            n_repeats = 10
            rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)
            scores = np.zeros((self.n_estimators, n_splits*n_repeats))

            for i in range(self.n_estimators):
                for fold_id, (train, test) in enumerate (rskf.split(X,y)):
                    clf = clone((self.base_estimator))
                    clf.fit(X[train], y[train])
                    y_pred = clf.predict(X[test])
                    scores[i, fold_id] = accuracy_score(y[test], y_pred)

            mean = np.mean(scores, axis=1)
            self.ranks = rankdata(mean, method='min')


        self.ensemble_ = []
        for i in range(self.n_estimators):
            self.X_ = []
            self.y_ = []

            for i in range(0, self.n_features):
                self.temp = random.randint(0, X.shape[0] - 1)
                self.X_.append(X[self.temp])
                self.y_.append(y[self.temp])

            self.X_ = np.array(self.X_)
            self.y_ = np.array(self.y_)
            self.ensemble_.append(clone(self.base_estimator).fit(self.X_, self.y_))
        return self

    def predict(self, X):
        # sprawdzenie czy modele są wyuczone
        check_is_fitted(self, "classes_")
        X = check_array(X)

        # głosowanie większościowe
        if self.hard_voting:
            pred_ = []

            if self.scales == True:
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X))

                pred_ = np.array(pred_)
                temp = []
                scores = np.zeros((pred_.T.shape[0], np.sum(self.ranks) + self.n_estimators))

                for k in range(0, pred_.T.shape[0]):
                    for i in range(0, self.n_estimators):
                        for j in range(0, self.ranks[i]):
                            temp.append(int(pred_.T[k][i]))

                    scores[k] = np.append(pred_.T[k], temp)
                    temp = []

                scores = scores.astype(int)
                prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=scores)

                return self.classes_[prediction]
            # do akumulacji wsparc
            else:
                for i, member_clf in enumerate(self.ensemble_):
                    pred_.append(member_clf.predict(X))

                pred_ = np.array(pred_)
                prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)

                return self.classes_[prediction]

        else:
            if self.scales == False:
                esm = self.ensemble_support_matrix(X)
                average_support = np.mean(esm, axis=0)
                prediction = np.argmax(average_support, axis=1)

                return self.classes_[prediction]
            else:
                average_support = self.ensemble_support_matrix(X)
                prediction = np.argmax(average_support, axis=1)

                return self.classes_[prediction]

    def ensemble_support_matrix(self, X):
        probas_ = []

        if self.scales == True:
            pred_ = []
            temp = []

            for i, member_clf in enumerate(self.ensemble_):
                pred_.append(member_clf.predict(X))

            pred_ = np.array(pred_)
            probas_ = np.array(probas_)
            classes = np.unique(pred_)
            n_classes = len(classes)
            probas_ = np.zeros((pred_.T.shape[0], n_classes))

            scores = np.zeros((pred_.T.shape[0], np.sum(self.ranks) + self.n_estimators))

            for k in range(0, pred_.T.shape[0]):
                for i in range(0, self.n_estimators):
                    for j in range(0, self.ranks[i]):
                        temp.append(int(pred_.T[k][i]))

                scores[k] = np.append(pred_.T[k], temp)
                temp = []
                length, counts = np.unique(scores[k], return_counts=True)

                temp_2 = np.zeros((n_classes))

                for j in range(0, n_classes):
                    if len(length) < n_classes:
                        for z in range(0, len(length)):
                            temp_2[z] = (counts[z] / np.sum(counts))
                    else:
                        temp_2[j] = (counts[j] / np.sum(counts))

                probas_[k] = temp_2

            return np.array(probas_)

        else:

            for i, member_clf in enumerate(self.ensemble_):
                probas_.append(member_clf.predict_proba(X))

            return np.array(probas_)

    # def predict(self, X):
    #     check_is_fitted(self, "classes_")
    #     X = check_array(X)
    #
    #     if self.hard_voting:
    #         pred_ = []
    #
    #         if self.scales == True:
    #             for i, member_clf in enumerate(self.ensemble_):
    #                 pred_.append(member_clf.predict(X))
    #
    #             pred_ = np.array(pred_)
    #             temp = []
    #             scores = np.zeros((pred_.T.shape[0], np.sum(self.ranks) + self.n_estimators))
    #
    #             for k in range(0, pred_.T.shape[0]):
    #                 for i in range(0, self.n_estimators):
    #                     for j in range(0, self.ranks[i]):
    #                         temp.append(int(pred_.T[k][i]))
    #
    #                 scores[k] = np.append(pred_.T[k], temp)
    #                 temp = []
    #
    #             scores = scores.astype(int)
    #             prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=scores)
    #
    #             return self.classes_[prediction]
    #         # do akumulacji wsparc
    #         else:
    #             for i, member_clf in enumerate(self.ensemble_):
    #                 pred_.append(member_clf.predict(X))
    #
    #             pred_ = np.array(pred_)
    #             prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=pred_.T)
    #
    #             return self.classes_[prediction]
    #
    #     else:
    #         if self.scales == False:
    #             esm = self.ensemble_support_matrix(X)
    #             average_support = np.mean(esm, axis=0)
    #             prediction = np.argmax(average_support, axis=1)
    #
    #             return self.classes_[prediction]
    #         else:
    #             average_support = self.ensemble_support_matrix(X)
    #             prediction = np.argmax(average_support, axis=1)
    #
    #             return self.classes_[prediction]
    #
    # def ensemble_support_matrix(self,X):
    #     probas_ =[]
    #     if self.scales ==True:
    #         pred_ =[]
    #         temp = []
    #
    #         for i, member_clf in enumerate(self.ensemble_):
    #             pred_.append(member_clf.predict(X))
    #
    #         pred_ = np.array(pred_)
    #         probas_ = np.array(probas_)
    #         classes = np.unique(pred_)
    #         n_classes = len(classes)
    #         probas_ = np.zeros((pred_.T.shape[0],n_classes))
    #
    #         scores = np.zeros((pred_.T.shape[0],np.sum(self.ranks) + self.n_estimators))
    #
    #         for k in range(0, pred_.T.shape[0]):
    #             for i in range(0,self.n_estimators):
    #                 for j in range(0,self.ranks[i]):
    #                     temp.append(int(pred_.T[k][i]))
    #
    #             scores[k] = np.append(pred_.T[k], temp)
    #             temp = []
    #             length, counts = np.unique(scores[k], return_counts=True)
    #
    #             temp_2 = np.zeros((n_classes))
    #
    #             for j in range(0, n_classes):
    #                 if len(length) < n_classes:
    #                     for z in range(0,len(length)):
    #                         temp_2[z] = (counts[z] / np.sum(counts))
    #                 else:
    #                     temp_2[j] = (counts[j] / np.sum(counts))
    #
    #             probas_[k] = temp_2
    #
    #         return np.array(probas_)
    #
    #     else:
    #         for i, member_clf in enumerate(self.ensemble_):
    #             probas_.append(member_clf.predict_proba(X))
    #         return np.array(probas_)