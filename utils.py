import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import pandas as pd

def load_data(model):
    """loads the data from the csv files, and converts to numpy arrays"""
    print('loading data ...')
    view = pd.read_csv('~/features/OR_2011_synthetic_' + model + '_features.csv', header=None, index_col=0)
    train_ids = pd.read_csv('~/split/OR_2011_train_IDs_synthetic.csv', header=0, index_col=0)
    test_ids = pd.read_csv('~/split/OR_2011_test_IDs_synthetic.csv', header=0, index_col=0)
    val_ids = pd.read_csv('~/split/OR_2011_val_IDs_synthetic.csv', header=0, index_col=0)
    
    train_set = make_tensor(view.loc[train_ids.x].values)
    test_set = make_tensor(view.loc[test_ids.x].values)
    val_set = make_tensor(view.loc[val_ids.x].values)
    dim = len(view.columns)

    return ([train_set, val_set, test_set], dim)


def make_tensor(data):
    """converts the input to numpy arrays"""
    data = torch.tensor(data)
    return data


def svm_classify(data, C):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """
    train_data, _, train_label = data[0]
    valid_data, _, valid_label = data[1]
    test_data, _, test_label = data[2]

    print('training SVM...')
    clf = svm.LinearSVC(C=C, dual=False)
    clf.fit(train_data, train_label.ravel())

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)
    p = clf.predict(valid_data)
    valid_acc = accuracy_score(valid_label, p)

    return [test_acc, valid_acc]


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret
