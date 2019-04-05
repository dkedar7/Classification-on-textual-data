import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from itertools import product
from sklearn.ensemble import VotingClassifier

class KNN_classifier(object):
    def __init__(self,param = 10):
        super().__init__()
        self.k = param

   def train(self,feature_matrix,label_vector):
        self.trained_model = KNN(n_neighbors = self.k).fit(feature_matrix, label_vector)

   def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)

class logreg_model(object):
    def __init__(self):
        super().__init__()

    def train(self,feature_matrix,label_vector):
        self.trained_model = LogisticRegression(random_state=1,
        solver="liblinear",multi_class="ovr",class_weight = "balanced",
        max_iter = 100).fit(feature_matrix,label_vector)

    def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)

class SVM(object):

    def __init__(self,C = 5):
        super().__init__()
        self.C = C

    def train(self,feature_matrix,label_vector):
        self.trained_model = SVC(C=self.C,class_weight="balanced",kernel="rbf",gamma="auto")
        self.trained_model.fit(feature_matrix,label_vector)

    def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)

class RandomForest(object):
    def __init__(self,ntrees,depth):
        super().__init__()
        self.ntrees = ntrees
        self.depth = depth

    def train(self,feature_matrix,label_vector):
        self.trained_model = RandomForestClassifier(n_estimators=self.ntrees, max_depth=self.depth,random_state=0)
        self.trained_model.fit(feature_matrix,label_vector)

    def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)

class MLP(object):
    def __init__(self,n_hidden,n_units):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_units = n_units

    def train(self,feature_matrix,label_vector):
        self.trained_model = MLPClassifier([self.n_units]*self.n_hidden,
        random_state = 2,max_iter = 500,warm_start = False)
        self.trained_model.fit(feature_matrix,label_vector)

    def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)

class AdaBoost(object):
    def __init__(self,n_estimators):
        super().__init__()
        self.n_estimators = n_estimators

    def train(self,feature_matrix,label_vector):
        self.trained_model = AdaBoostClassifier(n_estimators=self.n_estimators)
        self.trained_model.fit(feature_matrix,label_vector)

    def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)

class GNB(object):
    def __init__(self):
        super().__init__()

    def train(self,feature_matrix,label_vector):
        self.trained_model = GaussianNB()
        self.trained_model.fit(feature_matrix,label_vector)

    def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)

class Voting_Classifier(object):
    def __init__(self):
        super().__init__()

    def train(self,feature_matrix,label_vector):

        clf1 = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0)
        clf2 = MLPClassifier([200]*5,random_state = 2)
        clf3 = SVC(C=5,gamma="auto", kernel="rbf", probability=True)
        self.trained_model = VotingClassifier(estimators=[("mlp", clf2), ("svc", clf3)],
                                voting="soft", weights=[1, 1])
        self.trained_model.fit(feature_matrix,label_vector)

    def evaluate(self,feature_matrix):
        return self.trained_model.predict(feature_matrix)