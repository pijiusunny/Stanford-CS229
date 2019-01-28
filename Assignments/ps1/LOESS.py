#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 20:56:58 2018

@author: roy
"""

import numpy as np
from numpy.linalg import inv

def weight_construction(x, X, tau):
    '''
    Input:  K query points x, of dimension Kx1
            design matrix X, of dimension mx2
            tau, bandwidth parameter
    Output: m weights for each of the K query points, mxK dimensional matrix
    '''
    m = X.shape[0]
    K = x.shape[0]
    X1 = X[:,1]
    Weight = np.zeros((m,K))
    for i in range(K):
        weight = np.exp(-(x[i] - X1)**2 / (2*tau**2))
        Weight[:,i] = weight
    return Weight

def weighted_linear_reg(X, y, W):
    '''
    Input:  design matrix X, of dimension mx2
            vector of response variable y, of dimension mx1
            weight matrix W, of dimension mxm
    Output: fitted parameter vector theta, of dimension 2x1
    '''
    theta = np.matmul(inv(np.matmul(np.matmul(X.T, W), X)), np.matmul(np.matmul(X.T, W), y))
    return theta

def locally_weighted_linear_reg(x, X, y, tau):
    '''
    Input:  K query points x, of dimension Kx1
            design matrix X, of dimension mx2
            vector of response variable y, of dimension mx1
            tau, bandwidth parameter
    Output: fitted values at the K query points, of dimension Kx1
    '''
    K = x.shape[0]
    fitted_value = np.zeros((K,1))
    Weight = weight_construction(x, X, tau)
    for i in range(K):
        weight = Weight[:,i]
        W = 1/2 * np.diagflat(weight)
        theta = weighted_linear_reg(X, y, W)
        query_point = np.array((1, x[i]))
        query_point = query_point.reshape((2,1))
        fitted_value[i,0] = float(np.matmul(theta.T, query_point))
    return fitted_value

if __name__ == '__main__':
    pass