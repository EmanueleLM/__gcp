# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:26:13 2019

@author: Emanuele
"""
import numpy as np

def cumulative_distribution(Q, reshape=False):
    """
    Input is a tensor of any shape.
    """
    import copy as cp
    Q = cp.copy(Q)
    if reshape is True:
        Q_shape = Q.shape
    Q = Q.flatten()
    len_Q = len(Q)
    c_link_weights = np.zeros(shape=(len_Q))
    for (i,q) in zip(range(len_Q),Q.flatten()):
        c_link_weights[i] = np.count_nonzero(np.argwhere(q>Q))/len_Q
    if reshape == True:
        c_link_weights = c_link_weights.reshape(Q_shape)
    return c_link_weights