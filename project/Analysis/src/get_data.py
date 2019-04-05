import numpy as np
import sklearn
import os
import sys

from sklearn.datasets import fetch_20newsgroups
classes = ['comp.graphics','comp.os.ms-windows.misc',
'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware']

# Forming train and test data

def download():

    train = fetch_20newsgroups(data_home = "../Data", subset='train',
    categories=classes,
    shuffle=True,
    random_state=42,
    remove=('headers','footers','quotes'))


    test = fetch_20newsgroups(data_home = "../Data",
     subset='test',
     categories=classes,
     huffle=True,
     random_state=42,
     remove=('headers','footers','quotes'))

    train_text = [data['train'].data[i] for i in range(len(data['train'].data)) if \
    (data['train'].target_names)[data['train'].target[i]] in classes]

    train_labels = [(data['train'].target_names)[data['train'].target[i]] \
    for i in range(len(data['train'].data)) if \
    (data['train'].target_names)[data['train'].target[i]] in classes]

    test_text = [data['test'].data[i] for i in range(len(data['test'].data)) if \
    (data['test'].target_names)[data['test'].target[i]] in classes]

    test_labels = [(data['test'].target_names)[data['test'].target[i]] \
    for i in range(len(data['test'].data)) if \
    (data['test'].target_names)[data['test'].target[i]] in classes]


if __name__ == '__main__':
    download()
