# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:33:07 2019

@author: Emanuele
"""

def Qw(init_weights, fin_weights, num_params, dst):
    """
        Calculate the metric Q(w) and save it, for each layer, on file.
    """
    
    import numpy as np
    
    weights_shapes = [i.shape for i in init_weights]
    Qw_init, Qw_fin = np.zeros(shape=(num_params,)), np.zeros(shape=(num_params,))
    
    ctr_i, ctr_f = 0, 0
    for i in range(len(init_weights)):
        
        tmp1 = init_weights[i].flatten()
        tmp2 = fin_weights[i].flatten()
        len_w = len(tmp1)
        offset = 0
        
        print("[CUSTOM-LOGGER]: Percentage of same-value weights (init) {}.".format(len(np.unique(tmp1))/len(tmp1)))
        print("[CUSTOM-LOGGER]: Percentage of same-value weights (fin) {}.".format(len(np.unique(tmp2))/len(tmp2)))
        
        print("[CUSTOM-LOGGER]: Processing init-vector {} out of {}.".format(i, len(init_weights)))
        for j in range(len_w):
            Qw_init[ctr_i] = len(tmp1[tmp1[j]>tmp1])/len_w
            ctr_i += 1
            
        print("[CUSTOM-LOGGER]: Processing fin-vector {} out of {}.".format(i, len(init_weights)))
        for j in range(len_w):
            Qw_fin[ctr_f] = len(tmp2[tmp2[j]>tmp2])/len_w
            ctr_f += 1
                            
        assert ctr_i == np.sum([np.prod(w.shape) for w in init_weights[:i+1]]), "Mismatch between indexing and real size of the matrices: {} but expected is {}".format(ctr_i, np.sum([np.prod(w.shape) for w in init_weights[:i+1]]))
        assert ctr_f == np.sum([np.prod(w.shape) for w in init_weights[:i+1]]), "Mismatch between indexing and real size of the matrices: {} but expected is {}".format(ctr_i, np.sum([np.prod(w.shape) for w in init_weights[:i+1]]))
        
    # reshape Qw_init and Qw_fin to the nn layers'shapes
    offset = 0
    tmp1, tmp2 = [], []
    for i in range(len(init_weights)):
        tmp1.append(Qw_init[offset:offset+np.prod(weights_shapes[i])].reshape(*weights_shapes[i]))
        tmp2.append(Qw_fin[offset:offset+np.prod(weights_shapes[i])].reshape(*weights_shapes[i]))
        offset += np.prod(weights_shapes[i])
        
    Qw_init = np.asarray(tmp1)
    Qw_fin = np.asarray(tmp2)
        
    np.save(dst + 'init_Q_w.npy', Qw_init)
    np.save(dst + 'fin_Q_w.npy', Qw_fin)
    