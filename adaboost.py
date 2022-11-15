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
    weights = np.ones(len(X))/len(X)
    samples = [X,X]
    labels = [Y,Y]
    models = []
    alphas = []

    for iteration in range (0,max_iter):
        print(f"ITER: {iteration}")
        num_samples = len(samples[0])
        dt = tree.DecisionTreeClassifier(max_depth=1)
        dt = dt.fit(samples[0], labels[0]) # fit with *previous* data
        models.append(dt)
        predictions = dt.predict(samples[0]) # done on *previous* data
        correct = []
        for i, prediction in enumerate(predictions):
            correct.append(prediction == labels[0][i])
        num_incorrect = num_samples - sum(correct)
        
        epsilon = calc_epsilon(weights, correct)
        alpha = calc_alpha(epsilon)
        alphas.append(alpha)
        
        weights = update_weights(weights,alpha,correct)

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

def gen_weighted_dataset(samples, labels, correct_predicitons, w_right, w_wrong):
    n_right = int(100 * w_right)
    n_wrong = int(100 * w_wrong)
    new_samples = np.empty(samples.shape, int)
    new_labels = np.empty(labels.shape, int)
    weights = np.empty(0, float)
    for i, correct in enumerate(correct_predicitons):
        if correct == 1:
            new_samples = np.append(new_samples, [samples[i]]*n_right, axis=0)
            new_labels = np.append(new_labels, [labels[i]]*n_right, axis=0)
            weights = np.append(weights, [w_right]*n_right)
        else:
            new_samples = np.append(new_samples, [samples[i]]*n_wrong, axis=0)
            new_labels = np.append(new_labels, [labels[i]]*n_wrong, axis=0)
            weights = np.append(weights, [w_wrong]*n_wrong)
    return new_samples, new_labels, weights


def adaboost_test(X,Y,f,alpha): # not tested yet
    """Returns accuracy from given adaboost-trained models
    
    Gets a predicted sign from each sample, and compares against the real label
    Calculates the number of correct out of total samples
    """

    num_correct = 0
    predictions = []

    for dt in f:
        predictions.append(dt.predict(X))

    # for i, sample in enumerate(X):
    #     adaboost_pred = 0
    #     for j, dt in enumerate(f):
    #         adaboost_pred += dt.predict(sample) * alpha[j]
    #     num_correct += adaboost_pred == Y[i]
    print(f"Alpha values: {alpha}")

    return num_correct / len(Y)