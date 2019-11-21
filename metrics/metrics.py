# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:39:46 2019

@author: Emanuele
"""

import numpy as np


def conv_output(k_shape, I_shape, s=(1,1), p=(0,0)):
    """
    Calculate size of convolution between (Ih,Iw) image and (kh,kw) kernel,
     with stride and padding.
    """
    kh, kw = k_shape[0], k_shape[1]
    Ih, Iw = I_shape[0], I_shape[1]
    Oh, Ow = int((Ih-kh+2*p[1])/s[0]) + 1,  int((Iw-kw+2*p[1])/s[1]) + 1
    return Oh, Ow    


def conv2adj2d(k, I_shape, s=(1,1), p=(0,0)):   
    """
    Compute from a single channel convolution its adjacency matrix.
    """
    kh, kw = k.shape[0], k.shape[1]
    Ih, Iw = I_shape[0], I_shape[1]
    Oh, Ow = conv_output((kh, kw), (Ih, Iw), s, p)
    """
    In general you don't need to build adj, you just need to compute
     the non-zero cells and eventually their indices
    """
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
            """
            In general you don't need to build adj, you just need to compute
             the non-zero cells and eventually their indices
            """
            adj[i,j+offh+offv:j+kw+offh+offv] = k[j,:]    
            offh += Iw-1
    return adj


def nodes_strength(adj):
    """
    Nodes strength from the adjacency matrix.
    Input is a (n_inputs, n_outputs) adjacency matrix.
    """
    return (np.sum(adj, axis=0), np.sum(adj, axis=1))    


def multichannel_nodes_strength(k, I_shape, s=(1,1), p=(0,0)):
    """
    Input is in the format (W, H, Channels, Filters)
    """
    kh, kw = k.shape[0], k.shape[1]
    channels, filters = k.shape[2], k.shape[3]
    Ih, Iw = I_shape[0], I_shape[1] 
    Oh, Ow = conv_output((kh, kw), (Ih, Iw))
    s_in = np.zeros(shape=(channels, filters, Ih*Iw))
    s_out = np.zeros(shape=(channels, filters, Oh*Ow))
    for f in range(filters):
        for c in range(channels):
            adj = conv2adj2d(k[:,:,c,f], I_shape, s, p)
            strength = nodes_strength(adj)
            s_out[c,f] += strength[1] 
            s_in[c,f] += strength[0]
    s_in = np.sum(s_in, axis=1).reshape(channels, Ih*Iw)
    s_out = np.sum(s_out, axis=0).reshape(filters, Oh*Ow)
    return s_in, s_out
    