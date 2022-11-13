'''
Author: Antonio Lang
Date: 13 November 2022
Description: 
'''

from sklearn import tree
import numpy as np
import math

'''
DecisionTreeClassifier(max_depth=1)
_.fit(X,Y)
_.predift([x,y])
'''

def adaboost_train(X,Y,max_iter):
    """Returns list of trained decision tree and alpha values
    """

    weights = np.ones(len(X))/len(X) # potentially unnecessary
    samples = []
    labels = []
    models = []
    alpha_vals = []

    for i in range (1,max_iter):
        dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=1)
        dt = dt.fit(samples, labels)
        models.append(dt)
        predictions = dt.predict(samples)
        num_incorrect = 0
        for i, prediction in enumerate(predictions):
            num_incorrect += prediction == samples[i]

        # calculate alpha
        # append alpha
        # create proportional weighting dataset



    '''
    algo makes a new dataset with a proportional number of samples
    - make copies of a particular sample to reach necessary weighting

    f: array of trained decision tree stumps
    alpha: 1D array of alpha values
    return: f, alpha
    '''
    return models, alpha_vals

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