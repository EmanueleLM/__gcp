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
                     the layer is dense.
    """
    
    # 1. DON'T FORGET THE BIASES!
    # 2. USE A REPRESENTATION FOR EACH COUPLE OF CONSECUTIVE ADJ MATRICES
    assert len(adj_matrices) == len(strides), "Each adj matrix should have an entry in the conv list (int or None)."
    len_params = len(adj_matrices)
    
    weights_matrices = {}
       
    for i in range(len_params):
        
        current_, next_ = adj_matrices[i], adj_matrices[i+1]
        
        if i == 0:  # initial layers needs to consider also the input nodes
            
            current_ = np.sum(current_, axis=-1)
            
            s = (input_shapes[1][0]*input_shapes[1][1],  # 21x21
                 input_shapes[0][0]*input_shapes[0][1],  # 84x84
                 input_shapes[0][2])  # 4 channels
            
            stride = strides[0]            
            weights_matrices['0'] = np.zeros(shape=s)
            print(weights_matrices['0'].shape)
            
            for n in range(s[2]):
                for i in range(s[1]):                        
                    for j in range(current_.shape[0]):
                        
                        print(i,j)
                        offset = i*stride + j*input_shapes[0][0] + j*current_.shape[0]  # stride plus elements plus padding (to go to the next line) 
                        print("-->", offset,offset+current_.shape[0])
                        weights_matrices['0'][i,offset:offset+current_.shape[0],n] = current_[j,:,n]
                            
        elif i == len_params-1:  # last layer needs the output nodes also to be considered
            
            pass
            
        else:

            pass            
        
        pass
    
    return weights_matrices
    
    
def node_strength(matrix,
                  conv=None):
    """
       Return a list with the strength of input and output nodes as defined at 
       def. 10.3 chapter 10 Complex Networks by V. Latora et al.
       Please note that the input is the adjacency matrix, not the weights matrix!
    """
    if conv == None:
        
        input_strengths = np.sum(matrix, axis=0)
        output_strengths = np.sum(matrix, axis=1)
        nodes_str = {'s_in_layer': input_strengths,
                     's_out_layer': output_strengths}
    
    else:
        
        assert (conv >= 0 and type(conv) == int), "Conv parameter should be a positive integer"
        pass
    
    return nodes_str
