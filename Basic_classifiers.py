#!/usr/bin/python

#################################################
# CS5783: Machine Learning #
# Assignment 1: Basic Classifier #
# Problem 2: Constructing a simple data set #
# Problem 3: Linear classifier #
# Problem 4: Nearest neighbors classification #
# Sk. Mashfiqur Rahman #
#################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
# from prob_1 import KDTree

# Global data set for problem 2, 3, 4
N = 5000
mean_0 = [3, 2]
covariance_0 = [[5, 1], [1, 1]]
mean_1 = [8, 5]
covariance_1 = [[5, 0], [0, 2]]

X = np.concatenate((np.random.multivariate_normal(mean_0, covariance_0, N),
                    np.random.multivariate_normal(mean_1, covariance_1, N)), axis=0)
y = np.concatenate((np.zeros((N, 1), 'int64'),
                    np.ones((N, 1), 'int64')), axis=0)

mask = np.random.random(2*N) < 0.8
X_training = X[mask]
y_training = y[mask]
mask = np.logical_not(mask)
X_test = X[mask]
y_test = y[mask]


def prob_2():

    plt.figure(figsize=(16, 12))
    plt.plot(X[:N, 0], X[:N, 1], 'o', markerfacecolor='none', color='#75bbfd', label="class 0")
    plt.plot(X[N:, 0], X[N:, 1], 'o', markerfacecolor='none', color='#f97306', label="class 1")
    plt.xlabel('x1', fontsize=22)
    plt.ylabel('x2', fontsize=22)
    plt.suptitle("10000 random data points from a multivariate normal/Gaussian distributions.", fontsize=24)
    plt.legend(fontsize='22')
    plt.savefig('10000 random data points from a multivariate Gaussian distributions.png')
    plt.show()


def prob_3():

    beta = np.linalg.inv(X_training.T.dot(X_training)).dot(X_training.T).dot(y_training)
    y_hat = X_test.dot(beta)
    mask = X_test.dot(beta) < 0.5
    y_hat[mask] = 0
    mask = np.logical_not(mask)
    y_hat[mask] = 1

    c = np.count_nonzero(y_hat == y_test)  # count the number of true elements in Boolean array
    print('The classification accuracy of the algorithm is:', float(c / len(y_test))*100., '%')

    y_training_new = y_training.reshape(-1) # To form an 1D array
    y_test_new = y_test.reshape(-1)
    y_hat_new = y_hat.reshape(-1)

    training0 = X_training[y_training_new == 0]
    training1 = X_training[y_training_new == 1]
    correct0 = X_test[np.logical_and(y_test_new == 0, y_hat_new == 0)]
    correct1 = X_test[np.logical_and(y_test_new == 1, y_hat_new == 1)]
    incorrect0 = X_test[np.logical_and(y_test_new == 0, y_hat_new == 1)]
    incorrect1 = X_test[np.logical_and(y_test_new == 1, y_hat_new == 0)]

    plt.figure(figsize=(16, 12))
    plt.plot(training0[:, 0], training0[:, 1], 's', markerfacecolor='none', color='#75bbfd', label='Training set elements from class 0')
    plt.plot(training1[:, 0], training1[:, 1], 'x', color='#f97306', label='Training set elements from class 1')
    plt.plot(correct0[:, 0], correct0[:, 1], 'o', markerfacecolor='none', color='#00FF00', label='Correctly classified test set elements from class 0')
    plt.plot(correct1[:, 0], correct1[:, 1], '.', color='#800080', label='Correctly classified test set elements from class 1')
    plt.plot(incorrect0[:, 0], incorrect0[:, 1], '*', color='#EE82EE', label='Incorrectly classified test set elements from class 0')
    plt.plot(incorrect1[:, 0], incorrect1[:, 1], '+', color='k', label='Incorrectly classified test set elements from class 1')
    plt.xlabel('x1', fontsize=22)
    plt.ylabel('x2', fontsize=22)
    plt.suptitle("Linear Classifier performance map", fontsize=24)
    plt.legend()
    plt.savefig('Linear Classifier performance map.png')
    plt.show()


def prob_4():

    KDT = cKDTree(X_training).query(X_test, k=1)
    # KDT1 = KDTree(X_training).find_nearest(X_test)
    y_hat = y_training[KDT [1]]  # Ignoring the location of the neighbors (the second output array)
    # y_hat1 = y_training[KDT1 [1]]
    c = np.count_nonzero(y_hat == y_test)  # count the number of true elements in Boolean array
    # c1 = np.count_nonzero(y_hat1 == y_test)
    print('The classification accuracy of the KD tree classifier is:', float(c / len(y_test))*100., '%')
    # print('The classification accuracy of my own KD tree classifier is:', float(c1 / len(y_test))*100., '%')

    y_training_new = y_training.reshape(-1)
    y_test_new = y_test.reshape(-1)
    y_hat_new = y_hat.reshape(-1)

    training0 = X_training[y_training_new == 0]
    training1 = X_training[y_training_new == 1]
    correct0 = X_test[np.logical_and(y_test_new == 0, y_hat_new == 0)]
    correct1 = X_test[np.logical_and(y_test_new == 1, y_hat_new == 1)]
    incorrect0 = X_test[np.logical_and(y_test_new == 0, y_hat_new == 1)]
    incorrect1 = X_test[np.logical_and(y_test_new == 1, y_hat_new == 0)]

    plt.figure(figsize=(16, 12))
    plt.plot(training0[:, 0], training0[:, 1], 's', markerfacecolor='none', color='#75bbfd', label='Training set elements from class 0')
    plt.plot(training1[:, 0], training1[:, 1], 'x', color='#f97306', label='Training set elements from class 1')
    plt.plot(correct0[:, 0], correct0[:, 1], 'o', markerfacecolor='none', color='#00FF00', label='Correctly classified test set elements from class 0')
    plt.plot(correct1[:, 0], correct1[:, 1], '.', color='#800080', label='Correctly classified test set elements from class 1')
    plt.plot(incorrect0[:, 0], incorrect0[:, 1], '*', color='#EE82EE', label='Incorrectly classified test set elements from class 0')
    plt.plot(incorrect1[:, 0], incorrect1[:, 1], '+', color='k', label='Incorrectly classified test set elements from class 1')
    plt.xlabel('x1', fontsize=22)
    plt.ylabel('x2', fontsize=22)
    plt.suptitle("KD tree Classifier performance map", fontsize=24)
    plt.legend()
    plt.savefig('KD tree Classifier performance map.png')
    plt.show()


# Main Function
prob_2()   # Simple Data Set
prob_3()   # Linear Classifier
prob_4()   # KD tree Classifier
