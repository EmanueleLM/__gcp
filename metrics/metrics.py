# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:39:46 2019

@author: Emanuele
"""

import numpy as np


def conv_output(k_shape, I_shape, s=(1,1)):
    """
    Calculate size of convolution between (Ih,Iw) image and (kh,kw) kernel and stride.
    Padding is assumed to be integrated on the image.
    """
    kh, kw = k_shape[0], k_shape[1]
    Ih, Iw = I_shape[0], I_shape[1]
    Oh, Ow = int((Ih-kh)/s[0]) + 1,  int((Iw-kw)/s[1]) + 1
    return Oh, Ow    


def conv2adj2d(k, I_shape, s=(1,1), pad='VALID'):   
    """
    Compute from a single channel convolution its adjacency matrix.
    Input is a tensor of shape (W, H).
    k shape is (kheight, kwidth)
    """
    if k.ndim > 2 or len(I_shape) > 2:
        raise Exception("Dimensions Exception: input kernel and image must be bidimensional.")
    kh, kw = k.shape[0], k.shape[1]
    p = [[0,0],[0,0]]
    if pad == 'VALID':
        p = [[0,0],[0,0]]
    # add to the bottom/right the rows/columns if padding is odd
    elif pad == 'SAME':  
        if (I_shape[0]-kh)%s[0]!=0:
            total_pad_h = I_shape[0] - (I_shape[0]-kh)%s[0]
            if total_pad_h%2!=0:
                p[0][0] = int(total_pad_h/2)
                p[0][1] = p[0][0]+1
            else:
                p[0][0] = p[0][1] = int(total_pad_h/2)
        if (I_shape[1]-kw)%s[1]!=0:
            total_pad_w = I_shape[1] - (I_shape[1]-kw)%s[1]
            if total_pad_w%2!=0:
                p[1][0] = int(total_pad_w/2)
                p[1][1] = p[1][0]+1
            else:
                p[1][0] = p[1][1] = int(total_pad_w/2)
    else:
        raise Exception("Not Implemented Exception: {} id not an implemented padding strategy".format(p))
    Ih, Iw = I_shape[0]+p[0][0]+p[0][1], I_shape[1]+p[1][0]+p[1][1]
    Oh, Ow = conv_output((kh, kw), (Ih, Iw), s)
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


def multichannel_nodes_strength(k, I_shape, s=(1,1), pad='VALID'):
    """
    Input is a tensor of shape (W, H)
    k shape is (kheight, kwidth) or (kheight, kwidth, Channels, Filters)
    """
    kh, kw = k.shape[0], k.shape[1]
    if k.ndim == 2:
        k = k.reshape(kh, kw, 1, 1)
    p = [[0,0],[0,0]]
    if pad == 'VALID':
        p = [[0,0],[0,0]]
    # add to the bottom/right the rows/columns if padding is odd
    elif pad == 'SAME':  
        if (I_shape[0]-kh)%s[0]!=0:
            total_pad_h = I_shape[0] - (I_shape[0]-kh)%s[0]
            if total_pad_h%2!=0:
                p[0][0] = int(total_pad_h/2)
                p[0][1] = p[0][0]+1
            else:
                p[0][0] = p[0][1] = int(total_pad_h/2)
        if (I_shape[1]-kw)%s[1]!=0:
            total_pad_w = I_shape[1] - (I_shape[1]-kw)%s[1]
            if total_pad_w%2!=0:
                p[1][0] = int(total_pad_w/2)
                p[1][1] = p[1][0]+1
            else:
                p[1][0] = p[1][1] = int(total_pad_w/2)
    else:
        raise Exception("Not Implemented Exception: {} id not an implemented padding strategy".format(p))
    Ih, Iw = I_shape[0]+p[0][0]+p[0][1], I_shape[1]+p[1][0]+p[1][1]
    Oh, Ow = conv_output((kh, kw), (Ih, Iw), s)
    channels, filters = k.shape[2], k.shape[3]
    strength = np.zeros(shape=(channels, filters, Oh*Ow, Ih*Iw))
    for f in range(filters):
        for c in range(channels):
            strength[c,f,:,:] = conv2adj2d(k[:,:,c,f], I_shape, s=s, pad='VALID')
    s_in = np.sum(strength, axis=(1,2))
    s_out = np.sum(strength, axis=(0,3))
    return s_in, s_out

    