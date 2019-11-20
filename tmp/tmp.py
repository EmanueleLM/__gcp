# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:39:46 2019

@author: Emanuele
"""

import numpy as np


def conv2adj2d(k, i_shape, s=(1,1), p=(0,0)):    
    # kernel's, input's and output's height and width
    kh, kw = k.shape[0], k.shape[1]
    Ih, Iw = i_shape[0], i_shape[1]
    Oh, Ow = int((Ih-kh+2*p[1])/s[0]) + 1,  int((Iw-kw+2*p[1])/s[1]) + 1
    adj = np.zeros(shape=(Oh*Ow, Ih*Iw), dtype=np.float32)   
    #print(adj.shape)
    offv, offh = 0, 0  # vertical and horizontal offsets
    for i in range(adj.shape[0]):
        offh = 0
        if i%Ow == 0 and i>0:
            offv += s[1]*Iw
        offh = s[0]*(i%Ow)
        for j in range(kh):
            #print(i, j+offh+offv, j+kw+offh+offv)
            adj[i,j+offh+offv:j+kw+offh+offv] = k[j,:]    
            offh += Iw-1
    return adj
        