#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 17:12:58 2018

@author: roy
"""

import numpy as np

# Helper function used to compute h_theta
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# In Newton's method, we need gradient vector and Hessian matrix at the current parameter vector in
# order to update the parameter vector in the next iteration.
# I also compute the value of cost function J, so as to monitor that J is decreasing and converging
# to some limiting value with more iterations.
def costFunction(theta, X, y):
    '''
    Input:
    theta - nx1 parameter vector
    X - mxn data matrix
    y - mx1 label vector
    
    Output:
    J - value of cost function
    grad - gradient vector of cost function
    hess - Hessian matrix of cost function
    
    '''
    m = len(y)
    
    # Compute a useful (probability-like) quantity that appears in J, grad and hess
    prob = sigmoid(np.matmul(X,theta) * y)
    
    # Vectorized expressions of cost, gradient, Hessian
    # Cost
    J = float(-1/m * sum(np.log(prob)))
    
    # Gradient
    grad = -1/m * np.matmul(X.T, (y * (1 - prob)))
    
    # Hessian
    # Such vectorized computation using np.diagflat() may not be very efficient (both time and space)
    # I guess there are efficient implementations using 3D arrays/tensors, will look into it
    diagonal_matrix = np.diagflat(prob * (1 - prob))
    hess = 1/m * np.matmul(np.matmul(X.T,diagonal_matrix), X)
    
    return J, grad, hess

if __name__ == '__main__':
    pass

