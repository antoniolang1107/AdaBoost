'''
Author: Antonio Lang
Date: 15 November 2022
Description: Implements AdaBoost iterations and reweighting
'''

from sklearn import tree
import numpy as np
import math

''' Allowed sklearn.tree functions
DecisionTreeClassifier(max_depth=1)
_.fit(X,Y)
_.predict([[x,y]])
'''

def adaboost_train(X,Y,max_iter):
    """Returns list of trained decision tree and alpha values
    """

    weights = np.ones(len(X))/len(X) # potentially unnecessary
    samples = X
    labels = Y
    models = []
    alpha_vals = []

    for i in range (1,max_iter):
        num_samples = len(samples)
        dt = tree.DecisionTreeClassifier(max_depth=1)
        dt = dt.fit(samples, labels) # fit with *previous* data
        models.append(dt)
        predictions = dt.predict(samples) # done on *previous* data
        num_incorrect = 0
        for i, prediction in enumerate(predictions):
            num_incorrect += prediction == labels[i]
        
        alpha = calc_alpha(num_incorrect, num_samples)
        w_right, w_wrong = calc_correctness_weights(alpha, num_samples)
        z = w_wrong*num_incorrect+w_right*(num_samples-num_incorrect)
        w_right *= 1/z
        w_wrong *= 1/z




    '''
    algo makes a new dataset with a proportional number of samples
    - make copies of a particular sample to reach necessary weighting

    f: array of trained decision tree stumps
    alpha: 1D array of alpha values
    return: f, alpha
    '''
    return models, alpha_vals

def calc_alpha(num_incorrect, size_samples):
    epsilon = num_incorrect / size_samples
    alpha = 0.5 * math.log((1-epsilon)/epsilon)
    return alpha

def calc_correctness_weights(alpha, size_samples):
    w_right = 1/size_samples * math.exp(-1*alpha)
    w_wrong = 1/size_samples * math.exp(alpha)
    return w_right, w_wrong

def gen_weighted_dataset():
    pass

def adaboost_test(X,Y,f,alpha): # not tested yet
    """Returns accuracy from given adaboost-trained models
    
    Gets a predicted sign from each sample, and compares against the real label
    Calculates the number of correct out of total samples
    """

    num_correct = 0

    for i, sample in enumerate(X):
        adaboost_pred = 0
        for j, dt in enumerate(f):
            adaboost_pred += dt.predict(sample) * alpha[j]
        num_correct += adaboost_pred == Y[i]

    return num_correct / len(Y)