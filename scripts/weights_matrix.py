# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:23:37 2019

@author: Emanuele
"""

import numpy as np


def get_weights_matrix(adj_matrices,
                       biases,
                       strides,
                       input_shapes):
    """
        Generate the weights matrix, given a series of adjacency matrices. For a formal
        definition of weights matrix, see def. 10.1, 10.2 chapter 10 Complex Networks
        by V. Latora et al.
        adj_matrices:list of parameters, excluded biases;
        biases:list, list of biases, an entry can be None if bias for a specific layer is not present;
        strides:list, specifies strides for each convolutional layer. An entry is None if
                     the layer is dense; 
        input_shapes:list, list of input shapes i.e. the number of nodes in each layer. In our case
                           the right vector is [(84,84,4), (21,21,16), (11,11,32), (3872), (256), (18)]
                           even if the last three are not used (just the convs are needed).
    
    Speedup:
        1) load adj_matrix into variable (x = np.load(adj_filename, allow_pickle=True))
        2) adj_matrices = [fin_weights[0], fin_weights[2], fin_weights[4], fin_weights[6]] 
        3) biases = [fin_weights[1], fin_weights[3], fin_weights[5], fin_weights[7]]
        4) strides = [4,2, None, None]
        5) input_shapes = [(84,84,4), (21,21,16), (11,11,32), (3872), (256), (18)]
    """
    
    assert len(adj_matrices) == len(strides), "Each adj matrix should have an entry in the conv list (int or None)."
    len_params = len(adj_matrices)
    
    weights_strengths = {}
       
    for i in range(len_params):
               
        # first convolutional layer
        if i == 0:  # in the initial layers we need to consider also the input nodes
            
            # input strengths
            # reduce last dimension as it represents the number of filters
            current_ = np.sum(adj_matrices[i], axis=-1)  # 8x8x4x16 -> 8x8x4
            
            s = (input_shapes[1][0]*input_shapes[1][1],  # 21x21, output
                 input_shapes[0][0]*input_shapes[0][1],  # 84x84, input
                 input_shapes[0][2])  # 4 channels
            
            stride = strides[0]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                for m in range(s[0]):                        
                    for j in range(current_.shape[0]):
                        
                        offset = m*stride + j*input_shapes[0][0]
                        weights_matrices[m,offset:offset+current_.shape[0],n] = current_[j,:,n]
            
            weights_strengths['o-l0'] = np.sum(weights_matrices, axis=0).flatten()

            # output strengths
            # reduce semi-last dimension as it represents the number of channels
            current_ = np.sum(adj_matrices[i], axis=-2)
            
            s = (input_shapes[1][0]*input_shapes[1][1],  # 21x21, output
                 input_shapes[0][0]*input_shapes[0][1],  # 84x84, input
                 input_shapes[1][2])  # 16 filters
            
            stride = strides[0]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                for m in range(s[0]):                        
                    for j in range(current_.shape[0]):
                        
                        offset = m*stride + j*input_shapes[0][0]  # stride plus elements plus padding (to go to the next line) 
                        weights_matrices[m,offset:offset+current_.shape[0],n] = current_[j,:,n]
   
            if len(biases) >= i:                
                weights_matrices = np.sum([weights_matrices, biases[i]], axis=-1)[0]
                       
            weights_strengths['i-l1'] = np.sum(weights_matrices, axis=1).flatten()
                                
        # second convolutional layer
        if i == 1:
            
            # input strength
            # reduce last dimension as it represents the number of kernels
            current_ = np.sum(adj_matrices[i], axis=-1)
            
            s = (input_shapes[2][0]*input_shapes[2][1],  # 11x11, output
                 input_shapes[1][0]*input_shapes[1][1],  # 21x21, input
                 input_shapes[1][2])  # 16 channels
                                   
            stride = strides[1]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                for m in range(s[0]):                        
                    for j in range(current_.shape[0]):
                        
                        offset = m*stride + j*input_shapes[1][0]  # stride plus elements plus padding (to go to the next line) 
                        weights_matrices[m,offset:offset+current_.shape[0],n] = current_[j,:,n]           
                        
            weights_strengths['o-l1'] = np.sum(weights_matrices, axis=0).flatten()
            
            # output strengths
            # reduce semi-last dimension as it represents the number of channels
            current_ = np.sum(adj_matrices[i], axis=-2)
            
            s = (input_shapes[2][0]*input_shapes[2][1],  # 11x11, output
                 input_shapes[0][0]*input_shapes[1][1],  # 21x21, input
                 input_shapes[2][2])  # 32 filters
            
            stride = strides[0]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                for m in range(s[0]):                        
                    for j in range(current_.shape[0]):
                        
                        offset = m*stride + j*input_shapes[0][0]  # stride plus elements plus padding (to go to the next line) 
                        weights_matrices[m,offset:offset+current_.shape[0],n] = current_[j,:,n]
            
            if len(biases) >= i:                
                weights_matrices = np.sum([weights_matrices, biases[i]], axis=-1)[0]
                
            weights_strengths['i-l2'] = np.sum(weights_matrices, axis=1).flatten()
            
        elif i == 2 or i == 3:  # dense layers are easy to manage
            
            # output current layer
            weights_strengths['o-l'+str(i)] = np.sum(adj_matrices[i], axis=1).flatten()
            # input current layer
            weights_strengths['i-l'+str(i+1)] = np.sum(adj_matrices[i], axis=0).flatten()

            # take into account output biases
            if len(biases) >= i:
                weights_strengths['i-l'+str(i+1)] += biases[i]
                
    return weights_strengths
