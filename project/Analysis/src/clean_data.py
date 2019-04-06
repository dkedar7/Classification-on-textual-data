import numpy as np
import os
import sys

from get_data import download

from sklearn.externals import joblib

try:
    data = np.load("../Data/scratch_data.npz")
except:
    print ("Data needs to be downloaded. Downloading ...")
    download()
    data = np.load("../Data/scratch_data.npz")

train_text,train_labels= data['arr_0'], data['arr_1']
test_text,test_labels = data['arr_2'], data['arr_3']

def preprocess_text(text):

    # Rule 1
    modified_text = " ".join(test_text[i].split("\n\n")[1:])

    # Rule 2
    if "--" in modified_text:
        modified_text =  " ".join(modified_text.split("--")[:-1])

    # Rule 3
    modified_text = " ".join([stemmer.stem(X) for X in \
    (re.split(punctuations,text))])

    # Rule 4
    modified_text = modified_test.replace('\n','').\
    replace('\t','').replace('\r','').replace("  "," ")

    return modified_text

def vectorize_text(text,word_dict):
    return
