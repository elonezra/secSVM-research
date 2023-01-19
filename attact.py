# -*- coding: utf-8 -*-


from __future__ import division
import argparse
import glob
import logging
import numpy as np
import os
import time
import sklearn
import torch
import torch.multiprocessing as mp
import traceback
import ujson as json
from itertools import repeat
from pprint import pformat
import pickle

import apg.evasion as evasion
import apg.extraction as extraction
import apg.inpatients as inpatients
import apg.models as models
import apg.utils as utils
from apg.settings import config
from apg.utils import blue, yellow, red, green
import pandas as pd



#### imports that i made  ####

from scipy import sparse
import random


mp = torch.multiprocessing.get_context('forkserver')

real_prediction = 0
num_classes = 53

def main():
    start=time.time()
    args = parse_args()

    utils.configure_logging(args.run_tag, args.debug)
    logging.info(blue('Loading data...'))
    
    
    #do attack foolDeep
    with open('classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    
    print("finished open the file")
    
    y_pred = model.clf.predict(model.X_test)
    count=sum(y_pred)
    total=len(y_pred)
    real_prediction = count/total
    x_test_benign, x_test_malicious, y_test_benign, y_test_malicious = separate_malicious_benign(model.y_test,model.y_test,model.X_test)
    
    x_test_malicious_copy = x_test_malicious.copy()
    y_test_malicious_copy = y_test_malicious.copy()
    
    pred_san = deepfool_attack(model, model.X_train, x_test_malicious_copy, y_test_malicious_copy,model.clf)
    
    x_test_malicious_copy = x_test_malicious.copy()
    y_test_malicious_copy = y_test_malicious.copy()
    x_test_benign2, x_test_malicious, y_test_benign2, y_test_malicious = separate_malicious_benign(pred_san,y_test_malicious_copy,x_test_malicious_copy)
    x_test_benign, y_test_benign = connecting_apps(x_test_benign, x_test_benign2, y_test_benign, y_test_benign2)
    x_test_malicious_copy = x_test_malicious.copy()
    y_test_malicious_copy = y_test_malicious.copy()
    
    adv_predictions = add_nois_attack(model.clf,x_test_malicious_copy,y_test_malicious_copy)
    
    x_test_malicious_copy = x_test_malicious.copy()
    y_test_malicious_copy = y_test_malicious.copy()
    x_test_benign2, x_test_malicious, y_test_benign2, y_test_malicious = separate_malicious_benign(adv_predictions,y_test_malicious_copy,x_test_malicious_copy)
    x_test_benign, y_test_benign = connecting_apps(x_test_benign, x_test_benign2, y_test_benign, y_test_benign2)
    x_test_malicious_copy = x_test_malicious.copy()
    y_test_malicious_copy = y_test_malicious.copy()
    
    adv_predictions = cw_attack(model.clf,x_test_malicious_copy,y_test_malicious_copy)
    x_test_malicious_copy = x_test_malicious.copy()
    y_test_malicious_copy = y_test_malicious.copy()
    x_test_benign2, x_test_malicious, y_test_benign2, y_test_malicious = separate_malicious_benign(adv_predictions,y_test_malicious_copy,x_test_malicious_copy)
    x_test_benign, y_test_benign = connecting_apps(x_test_benign, x_test_benign2, y_test_benign, y_test_benign2)
    
    x_test_after_attact, y_test_after_attact = connecting_apps(x_test_benign, x_test_malicious, y_test_benign, y_test_malicious)
    evaluate_metrics(model.y_test,y_test_after_attact)
    
    
    
    t_np = x_test_after_attact.todense() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv("attact_featuer_set",index=False) #save to file
    
    exit(0)      



import numpy as np
 
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
def evaluate_metrics(y_pred, labels):
    # calculate evaluation metrics with sklearn
    accuracy = accuracy_score(labels, y_pred)
    recall = recall_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    # print evaluation metrics
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 score:", f1)
    print("_______________")
    
import numpy as np
from scipy.sparse import csr_matrix
def connecting_apps(x_test_1, x_test_2, y_test_1, y_test_2):
    for l in range(len(y_test_2)):
        x_test_1 = sparse.vstack((x_test_1,sparse.csr_matrix(x_test_2[l],shape=x_test_2[l].shape)))
        y_test_1 = np.append(y_test_1, np.int64(y_test_2[l]))
    return (x_test_1, y_test_1)

def separate_malicious_benign(labels, y_test, x_test):
    img = torch.empty((0,x_test.shape[1]), dtype=torch.int64)
    x_test_malicious = sparse.csr_matrix(img,shape=img.shape)
    x_test_benign = sparse.csr_matrix(img,shape=img.shape)
    y_test_malicious = np.array([])
    y_test_benign = np.array([])  
    for l in range(len(labels)):
        if labels[l] == 1:
            #malicious
            x_test_malicious = sparse.vstack((x_test_malicious,sparse.csr_matrix(x_test[l],shape=x_test[l].shape)))
            y_test_malicious = np.append(y_test_malicious, np.int64(labels[l]))
            
        elif labels[l] == 0:
            #benign"
            x_test_benign = sparse.vstack((x_test_benign,sparse.csr_matrix(x_test[l],shape=x_test[l].shape)))
            y_test_benign = np.append(y_test_benign, np.int64(labels[l]))
            
    print("amount of apps benign ",len(y_test_benign))
    print("amount of apps malicious ",len(y_test_malicious))
    return (x_test_benign, x_test_malicious, y_test_benign, y_test_malicious)
    
    
def division_size_applications(size):
    d1 = 1
    for i in range(2, 100):
        if size % i == 0:
            d1 = i
    return d1 
    
        
import numpy as np

def deepfool_attack(model, x_train, x_test, y_test, model2):
    x_test_copy = x_test.copy()
    y_test_copy = y_test.copy()
    d1 = division_size_applications(len(y_test))
    num_classes = d1
    q = d1
    img = torch.empty((0,x_test.shape[1]), dtype=torch.int64)
    attacked_image = sparse.csr_matrix(img,shape=img.shape)
    for t in range(int(len(y_test)/d1)):
    	h = d1 * t
    	x_test = x_test_copy.copy()
    	y_test = y_test_copy.copy()
    	x_test = x_test[h:h+d1]
    	y_test = y_test[h:h+d1]
    	adv_data = deepfool(model2, x_test,y_test,d1)
    	adv_data = csr_matrix(adv_data)
    	attacked_image = sparse.vstack((attacked_image,sparse.csr_matrix(adv_data,shape=adv_data.shape)))
    y_test = y_test_copy
    x_test = x_test_copy
    adv_predictions = model2.predict(attacked_image)
    evaluate_metrics(y_test,adv_predictions)
    return adv_predictions
        


import numpy as np
from scipy.sparse import csr_matrix
def gradients(model, x_2d,y_test,d1):
    x_2d_sparse = csr_matrix(x_2d)
    scores = model.decision_function(x_2d_sparse)
    x_2d_dense = x_2d_sparse.todense()
    x_2d_dense = x_2d_dense.reshape(d1, -1)
    y_test = y_test.reshape(d1, -1)
    gradients = model.coef_ * x_2d_dense.T * (1 - (y_test * scores).clip(-1, 1))
    return gradients



def deepfool(model, x, y_test,d1, num_classes=num_classes, overshoot=0.02, max_iter=5):
    r = np.zeros_like(x)
    original_preds = model.predict(x)
    current_class = np.argmax(original_preds)
    for _ in range(max_iter):
        x_d = x+r
        if type(x_d) == np.matrixlib.defmatrix.matrix:
            x_d = csr_matrix(x_d)
        x_dense = x_d.todense()
        grads = gradients(model, x_dense,y_test,d1)
        grads /= np.linalg.norm(grads)
        w = x + r
        if type(w) == np.matrixlib.defmatrix.matrix:
            w = csr_matrix(w)
        preds = model.predict(w)
        current_class = np.argmax(preds)
        i = 0
        r = np.zeros_like(grads)
        while current_class == np.argmax(preds):
            if r.shape != grads.shape:
                grads = grads.reshape(d1, -1)
            r += grads * (1+overshoot)
            i += 1
            if i > max_iter:
                break
            if r.shape != x.shape:
                r = r.T
            r = x + r
            r_sparse = csr_matrix(r)
            preds = model.predict(r_sparse)
    return r





def cw_attack(model, X_test, y_test):
    x_test_copy = X_test.copy()
    y_test_copy = y_test.copy()
    d1 = division_size_applications(len(y_test))
    num_classes = d1
    q = d1
    img = torch.empty((0,X_test.shape[1]), dtype=torch.int64)
    attacked_image = sparse.csr_matrix(img,shape=img.shape)
    for t in range(int(len(y_test)/d1)):
    	h = d1 * t
    	x_test = x_test_copy.copy()
    	y_test = y_test_copy.copy()
    	x_test = x_test[h:h+d1]
    	y_test = y_test[h:h+d1]
    	adv_data = cw_l2_attack(model,x_test,y_test,max_iter=500)
    	attacked_image = sparse.vstack((attacked_image,sparse.csr_matrix(adv_data,shape=adv_data.shape)))
    y_test = y_test_copy
    x_test = x_test_copy
    adv_predictions = model.predict(attacked_image)
    evaluate_metrics(y_test,adv_predictions)
    return adv_predictions



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from termcolor import cprint
# CW-L2 Attack
# Based on the paper, i.e. not exact same version of the code on https://github.com/carlini/nn_robust_attacks
# (1) Binary search method for c, (2) Optimization on tanh space, (3) Choosing method best l2 adversaries is NOT IN THIS CODE.
def cw_l2_attack(clf, images, labels, targeted=True, c=1e-4, kappa=0, max_iter=100, learning_rate=0.01) :
    images = images.toarray()  # convert the sparse matrix to a dense numpy array   
    labels = torch.from_numpy(labels)
    if labels.dtype == torch.float64:
        labels = torch.round(labels).to(torch.int64)
    labels = labels.to(device).reshape(-1)
    # Define f-function
    def f(x) :
        outputs = clf.model(x)
        one_hot_labels = torch.eye(len(outputs))[labels].to(device)
        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())
        
        # If targeted, optimize for making the other class most likely 
        if targeted :
            return torch.clamp(i-j, min=-kappa)  
        # If untargeted, optimize for making the other class most likely 
        else :
            return torch.clamp(j-i, min=-kappa)
        
    prev = 1e10
    float_tensor_cons = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    x = float_tensor_cons(images)
    w = torch.zeros_like(x, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate)
    for step in range(max_iter) :
        a = 1/2*(torch.tanh(w) + 1)
        loss1 = nn.MSELoss(reduction='sum')(a, x)
        loss2 = torch.sum(c*f(a))
        cost = loss1 + loss2
        optimizer.zero_grad(set_to_none=True)
        cost.backward(retain_graph=True)
        optimizer.step()
        # Early Stop when loss does not converge.
        if step % (max_iter//10) == 0 :
            if cost > prev :
                print('Attack Stopped due to CONVERGENCE....')
                return a
            prev = cost
        print('- Learning Progress : %2.2f %%        ' %((step+1)/max_iter*100), end='\r')
    
    attack_images = 1/2*(nn.Tanh()(w) + 1)
    return  sparse.csr_matrix(attack_images.detach().numpy())
    
    
import torchattacks
import torchvision.models as models

def add_nois_attack(model, x_test,y_test):
    noise_intensity = 0.1
    X_noisy = x_test 
    y_pred = model.predict(x_test)
    count=sum(y_pred)
    total=len(y_pred)
    real_prediction = count/total
    attack_prediction = real_prediction
    coef = sparse.csr.csr_matrix(model.coef_)
    X_noisy = X_noisy.multiply(coef)

    y_pred = model.predict(X_noisy)
    count=sum(y_pred)
    total=len(y_pred)
    attack_prediction = count/total
    evaluate_metrics(y_test,y_pred)
    return y_pred
 
 
    

def transplantation_wrapper(record, model, output_dir, args):
    """Wrapper to handle and debug errors from the problem space transplantation.

    Args:
        record (str): The path to the precomputed record.
        model (SVMModel): The `SVMModel` for the attack.
        output_dir (str): The root directory of where outputs should be stored.
        args (Namespace): Command line args.

    """
    logging.info('-' * 70)

    sha = utils.get_app_name(record)

    result = os.path.join(output_dir, 'success', 'report-{sha}.apk.json')
    if os.path.exists(result):
        logging.info(green('Already successfully generated!'))
        return

    failed = os.path.join(output_dir, 'failure', '{sha}.txt')
    if os.path.exists(failed) and not args.rerun_past_failures:
        logging.info(red('Already attempted to generate.'))
        return

    tries = config['tries']
    successful = False
    while not successful and tries > 0:
        try:
            evasion.problem_space_transplant(record, model, output_dir)
            successful = True
        except evasion.RetryableFailure as e:
            tries -= 1

            if tries > 0:
                logging.warning(red('Encountered a random error, retrying...'))
            else:
                logging.error(red('Ran out of tries :O Logging error...'))
                utils.log_failure(record, str(e), output_dir)
        except Exception as e:
            msg = 'Process fell over with: [{e}]: \n{traceback.format_exc()}'
            utils.log_failure(record, msg, output_dir)
            return e
    logging.info(yellow('Results in: {output_dir}'))





def parse_args():
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-R', '--run-tag', help='An identifier for this experimental setup/run.')
    p.add_argument('--confidence', default="25", help='The confidence level to use (%% of benign within margin).')
    p.add_argument('--n-features', type=int, default=None, help='Number of features to retain in feature selection.')
    p.add_argument('--max-permissions-per-organ', default=5, help='The number of permissions allowed per organ.')
    p.add_argument('--max-permissions-total', default=20, help='The total number of permissions allowed in an app.')

    # Stage toggles
    p.add_argument('-t', '--transplantation', action='store_true', help='Runs physical transplantation if True.')
    p.add_argument('--skip-feature-space', action='store_true',
                   help='Skips generation of patient records and feature estimates.')

    # Performance
    p.add_argument('--preload', action='store_true', help='Preload all host applications before the attack.')
    p.add_argument('--serial', action='store_true', help='Run the pipeline in serial rather than with multiprocessing.')

    # SecSVM hyperparameters
    p.add_argument('--secsvm', action='store_true')
    p.add_argument('--secsvm-k', default="0.25")
    p.add_argument('--secsvm-lr', default=0.0009, type=float)
    p.add_argument('--secsvm-batchsize', default=256, type=int)
    p.add_argument('--secsvm-nepochs', default=10, type=int)
    p.add_argument('--seed_model', default=None)

    # Harvesting options
    #p.add_argument('--harvest', action='store_true')
    #p.add_argument('--organ-depth', type=int, default=100)
    #p.add_argument('--donor-depth', type=int, default=10)

    # Misc
    p.add_argument('-D', '--debug', action='store_true', help='Display log output in console if True.')
    p.add_argument('--rerun-past-failures', action='store_true', help='Rerun all past logged failures.')

    args = p.parse_args()

    if args.secsvm_k == 'inf':
        args.secsvm_k = np.inf
    else:
        args.secsvm_k = float(args.secsvm_k)

    return args


if __name__ == "__main__":
    main()
