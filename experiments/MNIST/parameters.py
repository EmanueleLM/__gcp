# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:47:47 2019

@author: Emanuele
"""

import numpy as np

class SharedNoiseTable(object):
    
    def __init__(self, seed=123, count=250000000):
        
        import ctypes, multiprocessing

        # 1 gigabyte of 32-bit numbers. Will actually sample 2 gigabytes below.
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)  # 64-bit to 32-bit conversion here

    def get(self, i, dim):
        
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        
        return stream.randint(0, len(self.noise) - dim + 1)


def get_flat_parameters(seeds,
                        parameters=1008450,
                        seed=123,
                        count=250000000,
                        noise_stdev=5e-3,
                        save_to_file=False, 
                        dst='flat.npy'):
    """
       TODO: extract automatically also the initial seeds, now it relies on a .npy
        file taken directly from the real implementation.
       seeds:list, contains the seeds, as comma separated values, e.g. [1, 2, 3];
       ...
       dst:string, is the .npy file used to store the matrix.
    """
    
    number_seeds = len(seeds)
    assert number_seeds > 0
        
    noise = SharedNoiseTable(seed, count)  # initialize the noise matrix    
    v = np.zeros(shape=(number_seeds, parameters))
        
    for i in range(number_seeds):
        
        if i == 0:
            
            x = np.array([])
            tmp = np.load('initial_params/initial_params_' + str(seeds[0]) + '.npy', allow_pickle=True) 
            for t in tmp:
                
                x = np.concatenate((x, t.flatten()))
            
            v[i,:] = x
                                    
        else:
        
            v[i,:] = v[i-1,:] + noise_stdev * noise.get(seeds[i], parameters)
        
    if save_to_file:
        
        np.save(dst, v, allow_pickle=True)
        return v
        
    else:
        
        return v
    

def get_adjacency_matrices(model,
                           flat_vector,
                           save_to_file=False, 
                           dst='adj.npy'):
    
    """
       model:list, is a list of shapes you have to reduce the flat vector to. 
        Be careful that the flattened shape in the matrices must concide in shapes
        with the total number of parameters in the net e.g. a model with wieghts
        set to [(10,5), (3,7)] must have 10*5 + 3*7 parameters in the flattened vector;
       dst:string, is the .npy file used to store the matrix.
    """
        
    num_matrices = len(model)
    assert num_matrices > 0
    adj_matrices = np.array([np.zeros(shape=(s)) for s in model])
        
    ptr = 0
    for i in range(num_matrices):
        
        new_ptr = ptr + np.prod(model[i])
        adj_matrices[i] = flat_vector[ptr: new_ptr]
        adj_matrices[i] = adj_matrices[i].reshape(model[i])
        ptr = new_ptr
        
    if save_to_file is True:
        
        np.save(dst, adj_matrices, allow_pickle=True)
        return adj_matrices
        
    else:
        
        return adj_matrices
        