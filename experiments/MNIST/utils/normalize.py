# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:20:12 2019

@author: Emanuele
"""


def normalize_01(X1, X2=None, sequence=True, eps=1e-6):
    """
       Normalize each input between 0 and 1.
       If X1 is not None, it is assumed that the normalization has to be done between
        each element of X1 and X2, considering max and min from both the lists of matrices/tensors.
       If sequence is True, X1 (and eventually X2) are considered to be sequences (lists) of possibly different
        matrices/tensors, and each of those has to be normalized by its own (i.e. within its min and max values).
       Parameter eps is used to prevent division by zero issues.
       Returns two lists with both X1 and X2 normalized between 0 and 1.
    """
        
    import numpy as np

    if X2 is not None:
        assert len(X1) == len(X2)

    if sequence is False:
        X1, X2 = [X1], [X2]
    
    min_ = np.min([np.min([np.min(x1), np.min(x2)]) for (x1, x2) in zip(X1, X2)])        
    max_ = np.max([np.max([np.max(x1), np.max(x2)]) for (x1, x2) in zip(X1, X2)])
    
    if min_ <= 0.:
        max_ += np.abs(min_)
        
    range_ = range(len(X1))
    print("[CUSTOM-LOGGER] Min value is {},\n[CUSTOM-LOGGER] Max value is {}". format(min_, max_))
    
    for i in range_:
        
        if min_ <= 0.:            
            X1[i] += np.abs(min_); X2[i] += np.abs(min_)
        
        if max_ != 0.:
            X1[i] /= max_; X2[i] /= max_
        else:
            X1[i] /= eps; X2[i] /= eps
        
    return X1, X2
        