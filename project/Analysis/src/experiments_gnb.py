from time import time
start = time()

import numpy as np
import os

from tqdm import tqdm
import datetime

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import logging

from classifiers import *
from FeatureSelection import *

np.random.seed(0)

try:
    data = np.load("../Data/vectorized_data.npz")
except:
    os.system("python clean_data.py")
    data = np.load("../Data/vectorized_data")

train_tfidf,train_labels = data['arr_0'], data['arr_1']

# Import PC
train_PC = np.load("data_dump/train_data_tfidf_PC_2343.npy")

train_PC = train_tfidf

validation_PC = train_PC[-400:,:]
val_labels = train_labels[-400:]

train_PC = train_PC[:-400,:]
train_labels = train_labels[:-400]

# Experiment details
os.chdir("log/")

feature_selection = 'None'
algorithm = 'GNB'
runtime = 'Gaussian Naive Bayes with default parameters. All features used.\n \
Validation set has 400 points.'

log_name = 'GNB_'+str(datetime.datetime.now()).replace(' ','_')\
.replace(':','_').split(".")[0]+'.log'

logging.basicConfig(filename = log_name, level=logging.DEBUG)
logging.info('Feature_selection = ' + feature_selection)
logging.info('Algorithm = ' + algorithm)
logging.info('Runtime = ' + runtime)

# Experiemnt code
classifier = GNB()
classifier.train(train_PC,train_labels)

train_predict = classifier.predict(train_PC)
val_predict = classifier.predict(validation_PC)

tf1 = f1_score(train_labels,train_predict,average='macro')
vf1 = f1_score(val_labels,val_predict,average='macro')

logging.info('Train F1 = {} \t Validation F1 = {}'.format(tf1,vf1))

logging.info("Time to run : {} s.".format(time()-start))
