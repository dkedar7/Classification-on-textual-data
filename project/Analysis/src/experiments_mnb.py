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
algorithm = 'MNB'
runtime = 'Hyperparameter optimization in Gaussian Naive Bayes with alpha = np.linspace(0.01,0.1,50) default parameters. All features used.\n \
Validation set has 400 points.'

log_name = 'MNB_'+str(datetime.datetime.now()).replace(' ','_')\
.replace(':','_').split(".")[0]+'.log'

logging.basicConfig(filename = log_name, level=logging.DEBUG)
logging.info('Feature_selection = ' + feature_selection)
logging.info('Algorithm = ' + algorithm)
logging.info('Runtime = ' + runtime)

# Experiemnt code
train_f1 = []
val_f1 = []

alpha_range = np.linspace(0.01,0.1,50)

for alpha in tqdm(alpha_range):
    classifier = MNB(alpha = alpha)
    classifier.train(train_PC,train_labels)

    train_predict = classifier.predict(train_PC)
    val_predict = classifier.predict(validation_PC)

    tf1 = f1_score(train_labels,train_predict,average = 'macro')
    vf1 = f1_score(val_labels,val_predict,average = 'macro')

    train_f1.append(tf1)
    val_f1.append(vf1)

    logging.info('alpha = {} \t Train F1 = {} \t Validation F1 = {}'.format(alpha,tf1,vf1))

logging.info("Time to run : {} s.".format(time()-start))

plt.plot(alpha_range,train_f1,label='Train F1 Scores')
plt.plot(alpha_range,val_f1,label='Validation F1 Scores')
plt.xlabel("Alpha")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.savefig(log_name+'.png',bbox_inches = 'tight')
plt.show()
