#!/bin/bash

#3
classifiers='import numpy as np
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
        return self.trained_model.predict(feature_matrix)'

#4
feature_selection='import numpy as np
import pandas as pd

def VarianceThreshold(feature_matrix,threshold = 0.8):

    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=(threshold * (1 - threshold))).fit(feature_matrix)

    return selector

def selectKbest(feature_matrix,label_vector):

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.feature_selection import f_classif

    selector = SelectKBest(f_classif, k=80).fit(feature_matrix,label_vector)

    return selector

def RecursiveFeatureElimination(feature_matrix,label_vector,n_feat,kernel="rbf"):

    from sklearn.svm import SVC
    from sklearn.feature_selection import RFE

    svc = SVC(kernel=kernel, C=1)
    selector = RFE(estimator=svc, n_features_to_select=n_feat, step=1).fit(feature_matrix,label_vector)

    return selector

def PrincipalComponents(feature_matrix):

    from sklearn.decomposition import PCA
    selector = PCA().fit(feature_matrix)

    return selector'

#5
train_model='import numpy as np
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

   '
# Make folders and files

# root
mkdir Analysis
cd Analysis

# 1
mkdir Notebooks

# 1.1
mkdir Notebooks/images
touch Notebooks/ExploratoryDataAnalysis.ipynb
touch Notebooks/ModelEvaluation.ipynb

echo '{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\\n",
    "import sys\\n",
    "import subprocess"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required modules\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "csv_files = (subprocess.check_output(\"find ../Data/  -name \\\\*.csv\",shell=True)\\n",
    "            ).decode('"'utf-8'"').splitlines()\\n",
    "excel_files = (subprocess.check_output(\"find ../Data/  -name \\\\*.xlsx\",shell=True)\\n",
    "              ).decode('"'utf-8'"').splitlines()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "try:\\n",
    "    data = pd.read_csv(csv_files[0])\\n",
    "except:\\n",
    "    try:\\n",
    "        data = pd.read_csv(excel_files[0])\\n",
    "    except:\\n",
    "        print ('"'No .csv or .xlsx files found.'"')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# View data\\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count data\\n",
    "data.count()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Describe data\\n",
    "data.describe().transpose()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Quality checks"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dtypes of data\\n",
    "data.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Print columns whose dtypes are neither float not int\\n",
    "for column in data.columns:\\n",
    "    if ((data[column].dtype != int) and (data[column].dtype != float)):\\n",
    "        print (column)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exploratory Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}' > Notebooks/ExploratoryDataAnalysis.ipynb

echo '{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\\n",
    "import sys\\n",
    "import time\\n",
    "import subprocess"
   ]
 },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import required modules\\n",
    "\\n",
    "import numpy as np\\n",
    "import pandas as pd\\n",
    "from matplotlib import pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from tqdm import tqdm\\n",
    "\\n",
    "from sklearn.externals import joblib\\n",
    "from sklearn.preprocessing import StandardScaler\\n",
    "from sklearn.preprocessing import MinMaxScaler\\n",
    "\\n",
    "from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score\\n",
    "from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix\\n",
    "from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error"
   ]
 },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import saved models\\n",
    "model = joblib.load('"'../Models/SavedModels/'"')"
   ]
 },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pwd"
   ]
 }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
 },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
  },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.2"
 }
},
 "nbformat": 4,
 "nbformat_minor": 2
}' > Notebooks/ModelEvaluation.ipynb

# 2
mkdir src

# 2.1
touch src/FeatureSelection.py
touch src/classifiers.py
touch src/TrainModel.py
mkdir src/log

printf "$classifiers" > src/classifiers.py
printf "$feature_selection" > src/FeatureSelection.py
printf "$train_model" > src/TrainModel.py

# 3
mkdir Models
mkdir Models/AllModels
mkdir Models/SavedModels

# 4
mkdir Data
for file in ../*.*; do
  if [[ $file == *.csv ]]
    then
      cp $file Data/
  elif [[ $file == *.xlsx ]]
  then
    cp $file Data/
  fi;

done;

# Open and run jupyter notebook
cd Notebooks
jupyter nbconvert --to notebook --execute ExploratoryDataAnalysis.ipynb
jupyter notebook ExploratoryDataAnalysis.nbconvert.ipynb
