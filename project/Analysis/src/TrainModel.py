import numpy as np
import pandas as pd
import sys
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score,roc_curve

from model_evaluation_modi import kfold_cv,upsample
from feature_selection import VarianceThreshold, selectKbest, RecursiveFeatureElimination, PrincipalComponents

# Set arguments
model_type = sys.argv[1]
label_number = sys.argv[2]
feature_selector = sys.argv[3]

# feature_matrix =
# label_vector =

scale_data = True

if scale_data:
    scaler = StandardScaler(with_std=True).fit(feature_matrix)
    feature_matrix = scaler.transform(feature_matrix)

# Choose the model
if model_type == "lgr":
    from classifiers import logreg_model as model
    mod = model()

elif model_type == "knn":
    from classifiers import KNN_classifier as model
    param = 100
    mod = model(param = param)

elif model_type == "svm":
    from classifiers import SVM as model
    mod = model()

elif model_type == "rdf":
    from classifiers import RandomForest as model
    mod = model(ntrees = 5,depth = 2)

elif model_type == "mlp":
    from classifiers import MLP as model
    mod = model(n_hidden = 5,n_units = 200)

elif model_type == "adb":
    from classifiers import AdaBoost as model
    mod = model(n_estimators = 5)

elif model_type == "gnb":
    from classifiers import GNB as model
    mod = model()

elif model_type == "vclf":
    from classifiers import Voting_Classifier as model
    mod = model()

# Choose the feature select
if feature_selector == "vrt":
    selector = VarianceThreshold(feature_matrix,threshold = 0.8)[:,:]
    feature_matrix = selector.transform(feature_matrix)

elif feature_selector == "skb":
    selector = selectKbest(feature_matrix,label_vector)
    feature_matrix = selector.transform(feature_matrix)

elif feature_selector == "rfe":
    selector = RecursiveFeatureElimination(feature_matrix,label_vector,10,kernel="linear")
    feature_matrix = feature_matrix[:,np.array(selector.support_) == True]

elif feature_selector == "pca":
    selector = PrincipalComponents(feature_matrix)[:,:]
    feature_matrix = selector.transform(feature_matrix)

else:
    pass

start = time.time()
# Define and train model
mod = model().train(feature_matrix,label_vector) # Optional parameters
predictions = mod.evaluate(feature_matrix)

# Evaluate model

   