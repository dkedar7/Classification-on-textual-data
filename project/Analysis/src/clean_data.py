import numpy as np
import os
import sys

from get_data import download

from sklearn.externals import joblib

try:
    data = joblib.load("../Data/20news-bydate_py3.pkz")
except:
    print ("Data needs to be downloaded. Downloading ...")
    download()
    data = joblib.load("../Data/20news-bydate_py3.pkz")

train_data = data['train']
test_data = data['test']

def preprocess_text(text):

    modified_text = " ".join([stemmer.stem(X) for X in \
    (re.split(punctuations,text))])

    modified_text = modified_test.replace('\n','').\
    replace('\t','').replace('\r','').replace("  "," ")

    return modified_text
