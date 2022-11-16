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
    X = np.array(X)
    Y = np.array(Y)
    W = np.ones(len(X))/len(X)
    weights = [W,W]
    samples = [X,X]
    labels = [Y,Y]
    models = []
    alphas = []

    for iteration in range (0,max_iter):
        dt = tree.DecisionTreeClassifier(max_depth=1)
        dt = dt.fit(samples[0], labels[0]) # fit with *previous* data
        models.append(dt)
        predictions = dt.predict(samples[0]) # done on *previous* data
        correct = []
        for i, prediction in enumerate(predictions):
            correct.append(prediction == labels[0][i])
        
        epsilon = calc_epsilon(weights[0], correct)
        alpha = calc_alpha(epsilon)
        alphas.append(alpha)
        
        weights[0] = update_weights(weights[0],alpha,correct)
        samples[0] = samples[1].copy()
        labels[0] = labels[1].copy()
        weights[0] = weights[1].copy()
        samples[1], labels[1], weights[1] = gen_proportional_data(samples[0].copy(),labels[0].copy(),weights[0].copy())

    '''
    algo makes a new dataset with a proportional number of samples
    - make copies of a particular sample to reach necessary weighting

    f: array of trained decision tree stumps
    alpha: 1D array of alpha values
    return: f, alpha
    '''
    return models, alphas

def calc_epsilon(weights, correct):
    epsilon = 0
    for i in range(0,len(correct)-1):
        epsilon += weights[i] * (correct[i] == False)
    return epsilon

def calc_alpha(epsilon):
    alpha = 0.5 * math.log((1-epsilon)/epsilon)
    return alpha

def update_weights(weights, alpha, correct):
    for i, weight in enumerate(weights):
            weights[i] = update_weight(weight, alpha, correct[i])
    weight_sum = weights.sum()
    weights = weights/weight_sum
    return weights

def update_weight(weight, alpha, correctness):
    if correctness == 1:
        return weight * math.exp(-alpha)
    else:
        return weight * math.exp(alpha)

def gen_proportional_data(samples, labels, weights):
    scale = 100
    new_samples = np.empty([0,samples.shape[1]], int)
    new_labels = np.empty(0, int)
    new_weights = np.empty(0,float)
    for i in range(0,samples.shape[0]-1):
        new_samples = np.append(new_samples, np.array([samples[i]] * int(scale*weights[i])), axis=0)
        new_labels = np.append(new_labels, np.array([labels[i]] * int(scale*weights[i])))
        new_weights = np.append(new_weights, np.array([weights[i]/int(scale*weights[i])] * int(scale*weights[i])))
    return new_samples.copy(), new_labels.copy(), new_weights.copy()

def adaboost_test(X,Y,f,alpha): # not tested yet
    """Returns accuracy from given adaboost-trained models
    
    Gets a predicted sign from each sample, and compares against the real label
    Calculates the number of correct out of total samples
    """
    num_correct = 0
    predictions = []
    for dt in f:
        predictions.append(dt.predict(X))
    predictions = np.array(predictions)
    aggr_predictions = np.matmul(predictions.T,alpha)
    ada_predictions = np.sign(aggr_predictions)
    num_correct = np.equal(ada_predictions, Y).sum()

    print(f"Alpha values: {alpha}")
    return num_correct / len(Y)