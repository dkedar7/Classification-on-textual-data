import numpy as np
import os
import sys
import re

from get_data import download

from sklearn.externals import joblib
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

try:
    data = np.load("../Data/scratch_data.npz")
except:
    print ("Data needs to be downloaded. Downloading ...")
    download()
    data = np.load("../Data/scratch_data.npz")

train_text,train_labels = data['arr_0'], data['arr_1']
test_text,test_labels = data['arr_2'], data['arr_3']

def preprocess_text(text):

    # Rule 1
    modified_text = " ".join(text.split("\n\n")[1:])

    # Rule 2
    if "--" in modified_text:
        modified_text =  " ".join(modified_text.split("--")[:-1])

    # Rule 3
    stemmer = SnowballStemmer("english")
    punctuations = '[! \" # $ % \& \' \( \) \ * + , \- \. \/ : ; <=> ? @ \[ \\ \] ^ _ ` { \| } ~]'
    modified_text = " ".join([stemmer.stem(X) for X in \
    (re.split(punctuations,text))])

    # Rule 4
    modified_text = modified_text.replace('\n','').\
    replace('\t','').replace('\r','').replace("  "," ")

    return modified_text

def vectorize_text():

    train_text,train_labels= data['arr_0'], data['arr_1']
    test_text,test_labels = data['arr_2'], data['arr_3']

    train_text = [preprocess_text(text) for text in train_text]
    test_text = [preprocess_text(text) for text in test_text]

    np.savez("../Data/cleaned_data",train_text,train_labels,test_text,test_labels)

    vectorizer_count = CountVectorizer(stop_words ='english').fit(train_text)

    train_countvector = vectorizer_count.transform(train_text)
    test_countvector = vectorizer_count.transform(test_text)

    vectorizer_tfidf = TfidfTransformer().fit(train_countvector)

    train_tfidf = vectorizer_tfidf.transform(train_countvector)
    test_tfidf = vectorizer_tfidf.transform(test_countvector)

    return train_tfidf,test_tfidf

# Write to file
train_tfidf,test_tfidf = vectorize_text()
np.savez("../Data/vectorized_data",train_tfidf.toarray(),train_labels,test_tfidf.toarray(),test_labels)
