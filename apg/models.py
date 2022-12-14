# -*- coding: utf-8 -*-

"""
models.py
~~~~~~~~~

Available target models:
    * SVMModel - a base class for SVM-like models
        - SVM - Standard linear SVM using scikit-learn implementation
        - SecSVM - Secure SVM variant using a PyTorch implementation (based on [1])

[1] Yes, Machine Learning Can Be More Secure! [TDSC 2019]
    -- Demontis, Melis, Biggio, Maiorca, Arp, Rieck, Corona, Giacinto, Roli

"""
from  collections import OrderedDict
import logging
import numpy as np
import os
import pickle
import random
import ujson as json
from collections import OrderedDict
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import lib.secsvm
from apg.settings import config
from apg.utils import blue, red, yellow


class SVMModel:
    """Base class for SVM-like classifiers."""

    def __init__(self, X_filename, y_filename, num_features,X_test,y_test):

        self.X_filename = X_filename
        self.y_filename = y_filename
        self.X_test_filename = X_test
        self.y_test_filename = y_test
        #self.meta_filename = meta_filename
        self._num_features = num_features
        self.clf, self.vec = None, None
        self.column_idxs = []
        self.X_train, self.y_train, self.m_train = [], [], []
        self.X_test, self.y_test, self.m_test = [],[], []
        self.feature_weights, self.benign_weights, self.malicious_weights = [], [], []
        self.weight_dict = OrderedDict()

    def generate(self, save=True):
        """Load and fit data for new model."""
        logging.debug(blue('No saved models found, generating new model...'))
        X_train, y_train,self.vec,X_test,y_test  = load_features_new(self.X_filename, self.y_filename, self.X_test_filename,self.y_test_filename, True)
        self.column_idxs = self.perform_feature_selection(X_train, y_train)
        self.X_train = X_train[:, self.column_idxs]
        self.X_test = X_test[:, self.column_idxs]
        self.y_train = y_train
        self.y_test = y_test
        
        self.clf = self.fit(self.X_train, self.y_train)
        features = [self.vec.feature_names_[i] for i in self.column_idxs]

        w = self.get_feature_weights(features)
        self.feature_weights, self.benign_weights, self.malicious_weights = w
        self.weight_dict = OrderedDict(
            (w[0], w[2]) for w in self.feature_weights)

        if save:
            self.save_to_file()

    def dict_to_feature_vector(self, d):
        """Generate feature vector given feature dict."""
        return self.vec.transform(d)[:, self.column_idxs]

    def get_feature_weights(self, feature_names):
        """Return a list of features ordered by weight.

        Each feature has it's own 'weight' learnt by the classifier.
        The sign of the weight determines which class it's associated
        with and the magnitude of the weight describes how influential
        it is in identifying an object as a member of that class.

        Here we get all the weights, associate them with their names and
        their original index (so we can map them back to the feature
        representation of apps later) and sort them from most influential
        benign features (most negative) to most influential malicious
        features (most positive). By default, only negative features
        are returned.

        Args:
            feature_names: An ordered list of feature names corresponding to cols.

        Returns:
            list, list, list: List of weight pairs, benign features, and malicious features.

        """
        assert self.clf.coef_[0].shape[0] == len(feature_names)

        coefs = self.clf.coef_[0]
        weights = list(zip(feature_names, range(len(coefs)), coefs))
        weights = sorted(weights, key=lambda row: row[-1])

        # Ignore 0 weights
        benign = [x for x in weights if x[-1] < 0]
        malicious = [x for x in weights if x[-1] > 0][::-1]
        return weights, benign, malicious

    def perform_feature_selection(self, X_train, y_train):
        """Perform L2-penalty feature selection."""
        if self._num_features is not None:
            logging.info(red('Performing L2-penalty feature selection'))
            selector = LinearSVC(C=1)
            selector.fit(X_train, y_train)

            cols = np.argsort(np.abs(selector.coef_[0]))[::-1]
            #print(self._num_features,len(cols))
            cols = cols[:self._num_features]
        else:
            cols = [i for i in range(X_train.shape[1])]
        return cols

    def save_to_file(self):
        with open(self.model_name, 'wb') as f:
            pickle.dump(self, f)


class SVM(SVMModel):
    """Standard linear SVM using scikit-learn implementation."""

    def __init__(self, X_filename, y_filename, X_test,y_test,num_features=None):
        super().__init__(X_filename, y_filename, num_features,X_test,y_test)
        self.model_name = self.generate_model_name()

    def fit(self, X_train, y_train):
        logging.debug(blue('Creating model'))
        clf = LinearSVC(C=1)
        clf.fit(X_train, y_train)
        return clf

    def generate_model_name(self):
        model_name = 'svm'
        model_name += '.p' if self._num_features is None else '-f{}.p'.format(self._num_features)
        return os.path.join(config['models'], model_name)


class SecSVM(SVMModel):
    """Secure SVM variant using a PyTorch implementation."""

    def __init__(self, X_filename, y_filename,X_test,y_test, num_features=None,
                 secsvm_k=0.2, secsvm=False, secsvm_lr=0.0001,
                 secsvm_batchsize=1024, secsvm_nepochs=75, seed_model=None):
        super().__init__(X_filename, y_filename, num_features, X_test,y_test)
        self._secsvm = secsvm
        self._secsvm_params = {
            'batchsize': secsvm_batchsize,
            'nepochs': secsvm_nepochs,
            'lr': secsvm_lr,
            'k': secsvm_k
        }
        self._seed_model = seed_model
        self.model_name = self.generate_model_name()
        
    def fit(self, X_train, y_train):
        logging.debug(blue('Creating model'))
        clf = lib.secsvm.SecSVM(lr=self._secsvm_params['lr'],
                                batchsize=self._secsvm_params['batchsize'],
                                n_epochs=self._secsvm_params['nepochs'],
                                K=self._secsvm_params['k'],
                                seed_model=self._seed_model)
        clf.fit(X_train, y_train)
        return clf


    def generate_model_name(self):
        model_name = 'secsvm-k{}-lr{}-bs{}-e{}'.format(
            self._secsvm_params['k'],
            self._secsvm_params['lr'],
            self._secsvm_params['batchsize'],
            self._secsvm_params['nepochs'])
        if self._seed_model is not None:
            model_name += '-seeded'
        model_name += '.p' if self._num_features is None else '-f{}.p'.format(self._num_features)
        return os.path.join(config['models'], model_name)


def load_from_file(model_filename):
    logging.debug(blue('Loading model from {model_filename}...'))
    with open(model_filename, 'rb') as f:
        return pickle.load(f)


def load_features_new(X_train, y_train, X_test,y_test, load_indices=True):
    
    with open(X_train, 'rt') as f:
        X_tr = json.load(f)
    with open(y_train, 'rt') as f:
        y_tr = json.load(f)
     
    with open(X_test, 'rt') as f:
        X_te = json.load(f)
    
    with open(y_test, 'rt') as f:
        y_te = json.load(f)
    X_conc=X_tr+X_te
    y_conc=y_tr+y_te
    train_test_sp=len(y_tr)+1
    X, y, vec = vectorize(X_conc, y_conc)
    return X[:train_test_sp], y[:train_test_sp],vec,X[train_test_sp:],y[train_test_sp:]



def vectorize(X, y):
    vec = DictVectorizer()
    X = vec.fit_transform(X)
    y = np.asarray(y)
    return X, y, vec
'''
def load_features(X_filename, y_filename, load_indices):
    with open(X_filename, 'rt') as f:
        X = json.load(f)
        #[o.pop('sha256') for o in X]  # prune the sha, uncomment if needed
    with open(y_filename, 'rt') as f:
        y = json.load(f)
        # y = [x[0] for x in json.load(f)]  # prune the sha, uncomment if needed
    #with open(meta_filename, 'rt') as f:
    #    meta = json.load(f)

    X, y, vec = vectorize(X, y)
    print(X[1])
    print("train",X.shape[1])
    return X, y,vec

def load_features_test(X_filename, y_filename,vec, load_indices=True):
    with open(X_filename, 'rt') as f:
        X = json.load(f)
        #[o.pop('sha256') for o in X]  # prune the sha, uncomment if needed
    with open(y_filename, 'rt') as f:
        y = json.load(f)
        # y = [x[0] for x in json.load(f)]  # prune the sha, uncomment if needed
    #with open(meta_filename, 'rt') as f:
    #    meta = json.load(f)
    
    X = vec.fit_transform(X)
    y = np.asarray(y)
    print("test",X.shape[1])
    return X, y

'''
