# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:33:07 2019

@author: Emanuele
"""

def Qw(init_weights, fin_weights, dst):
    """
        Calculate the metric Q(w) and save it, for each layer, on file.
    """
    
    import numpy as np
    
    weights_shapes = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]    
    Qw_init, Qw_fin = [], []
    
    for i in range(len(init_weights)):
        
        print("[CUSTOM-LOGGER]: Processing vector {} out of {}.".format(i, len(init_weights)))
        tmp1 = init_weights[i].flatten()
        tmp2 = fin_weights[i].flatten()
        len_w = len(tmp1)
        for j in range(len_w):
            Qw_init.append(len(tmp1[tmp1>tmp1[j]])/len_w)
            Qw_fin.append(len(tmp2[tmp2>tmp2[j]])/len_w)
            
    Qw_init = np.asarray(Qw_init)
    Qw_fin = np.asarray(Qw_fin)
    
    # reshape Qw_init and Qw_fin to the nn layers'shapes
    offset = 0
    tmp1, tmp2 = [], []
    for i in range(8):
        tmp1.append(Qw_init[offset:offset+np.prod(weights_shapes[i])].reshape(*weights_shapes[i]))
        tmp2.append(Qw_fin[offset:offset+np.prod(weights_shapes[i])].reshape(*weights_shapes[i]))
        offset += np.prod(weights_shapes[i])
        
    Qw_init = np.asarray(tmp1)
    Qw_fin = np.asarray(tmp2)
        
    np.save(dst + 'init_Q_w.npy', Qw_init)
    np.save(dst + 'fin_Q_w.npy', Qw_fin)
    