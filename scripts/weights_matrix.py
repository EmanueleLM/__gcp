# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:23:37 2019

@author: Emanuele
"""

import numpy as np


def get_weights_matrix(adj_matrices,
                       biases,
                       strides,
                       input_shapes,
                       input_padded,
                       transform=None):
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
                           even if the last three are not used (just the convs are needed);
        input_padded:list, input shapes when padded;
        transform:function, specify a function that is used to process each weight before addin it up to the weights strengths.
                            It can be used to calculate metrics as Node disparity, def. 10.4 Chapter 10 Complex Networks book,
                            by specifying a lambda function like sq=lambda x: x**2 and passing it as argument to transform.
    Speedup init/final:
        adj_matrices = [fin_weights[0], fin_weights[2], fin_weights[4], fin_weights[6]]
        biases = [fin_weights[1], fin_weights[3], fin_weights[5], fin_weights[7]]
        strides = [4,2, None, None]
        input_shapes = [(84,84,4), (21,21,16), (11,11,32), (3872), (256), (18)]
        input_padded = [(89,89,4), (25,25,16), (11,11,32), (3872), (256), (18)]
        fin = get_weights_matrix(adj_matrices,
                               biases,
                               strides,
                               input_shapes,
                               input_padded,
                               transform=None)
    """
    
    assert len(adj_matrices) == len(strides), "Each adj matrix should have an entry in the conv list (int or None)."
    len_params = len(adj_matrices)
    
    weights_strengths = {}
       
    for i in range(len_params):
        
        if transform is not None:
            adj_matrices[i] = transform(adj_matrices[i])
            if len(biases) >= i:
                biases[i] = transform(biases[i])
               
        # first convolutional layer
        if i == 0:  # in the initial layers we need to consider also the input nodes
            
            # input strengths
            # reduce last dimension as it represents the number of filters
            current_ = np.sum(adj_matrices[i], axis=-1)  # 8x8x4x16 -> 8x8x4
            
            s = (input_shapes[1][0]*input_shapes[1][1],  # 21x21, output
                 input_padded[0][0]*input_padded[0][1],  # 89x89, input
                 input_shapes[0][2])  # 4 channels
            
            stride = strides[0]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                vertical_offset = 0
                for m in range(s[0]):
                    horizontal_offset = (m%(current_.shape[0]))*stride  # offset induced by striding horizontally
                    vertical_offset += (stride*input_padded[0][0] if (m%21)==0 and m>0 else 0)  # offset induced by striding vertically
                    offset_rows = horizontal_offset + vertical_offset
                    for j in range(current_.shape[0]):
                        offset_col = j*(current_.shape[0] + input_padded[0][0])  # offset for each entry in the 8x8 matrix
                        offset = offset_col + offset_rows
                        weights_matrices[m,offset:offset+current_.shape[0],n] = current_[j,:,n]
                        
            weights_strengths['o-l0'] = np.sum(weights_matrices, axis=0).flatten()

            # output strengths
            # reduce semi-last dimension as it represents the number of channels
            current_ = np.sum(adj_matrices[i], axis=-2)
            
            s = (input_shapes[1][0]*input_shapes[1][1],  # 21x21, output
                 input_padded[0][0]*input_padded[0][1],  # 89x89, input
                 input_shapes[1][2])  # 16 filters
            
            stride = strides[0]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                vertical_offset = 0
                for m in range(s[0]): 
                    horizontal_offset = (m%(current_.shape[0]))*stride  # offset induced by striding horizontally
                    vertical_offset += (stride*input_padded[0][0] if (m%21)==0 and m>0 else 0)  # offset induced by striding vertically
                    offset_rows = horizontal_offset + vertical_offset
                    for j in range(current_.shape[0]):                        
                        offset_col = j*(current_.shape[0] + input_padded[0][0])  # offset for each entry in the 8x8 matrix
                        offset = offset_col + offset_rows
                        weights_matrices[m,offset:offset+current_.shape[0],n] = current_[j,:,n]
   
            if len(biases) >= i:                
                weights_matrices = np.sum([weights_matrices, biases[i]], axis=-1)[0]
            
            # as the input to the next edge is padded after the convolution, we need to manually add some padding to
            #  the first 2 axis

            weights_matrices = weights_matrices.reshape(input_shapes[1][0], input_shapes[1][0], 
                                                        input_padded[0][0]*input_padded[0][0],
                                                        input_shapes[1][2])
            weights_strengths['i-l1'] = np.sum(weights_matrices, axis=2)
            weights_strengths['i-l1'] = np.pad(weights_strengths['i-l1'], ((2,2),(2,2),(0,0)), 'edge').flatten()
                                
        # second convolutional layer
        if i == 1:
            
            # input strength
            # reduce last dimension as it represents the number of kernels
            current_ = np.sum(adj_matrices[i], axis=-1)
            
            s = (input_shapes[2][0]*input_shapes[2][1],  # 11x11, output
                 input_padded[1][0]*input_padded[1][1],  # 25x25, input
                 input_shapes[1][2])  # 16 channels
                                   
            stride = strides[1]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                vertical_offset = 0
                for m in range(s[0]): 
                    horizontal_offset = (m%(current_.shape[0]))*stride  # offset induced by striding horizontally
                    vertical_offset += (stride*input_padded[1][0] if (m%11)==0 and m>0 else 0)  # offset induced by striding vertically
                    offset_rows = horizontal_offset + vertical_offset
                    for j in range(current_.shape[0]):                        
                        offset_col = j*(current_.shape[0] + input_padded[1][0])  # offset for each entry in the 8x8 matrix
                        offset = offset_col + offset_rows
                        weights_matrices[m,offset:offset+current_.shape[0],n] = current_[j,:,n]           
                        
            weights_strengths['o-l1'] = np.sum(weights_matrices, axis=0).flatten()
            
            # output strengths
            # reduce semi-last dimension as it represents the number of channels
            current_ = np.sum(adj_matrices[i], axis=-2)
            
            s = (input_shapes[2][0]*input_shapes[2][1],  # 11x11, output
                 input_padded[0][0]*input_padded[1][1],  # 25x25, input
                 input_shapes[2][2])  # 32 filters
            
            stride = strides[1]            
            weights_matrices = np.zeros(shape=s)
            
            for n in range(s[2]):
                vertical_offset = 0
                for m in range(s[0]):  
                    horizontal_offset = (m%(current_.shape[0]))*stride  # offset induced by striding horizontally
                    vertical_offset += (stride*input_padded[1][0] if (m%11)==0 and m>0 else 0)  # offset induced by striding vertically
                    for j in range(current_.shape[0]):                        
                        offset_col = j*(current_.shape[0] + input_padded[1][0])  # offset for each entry in the 4x4 matrix
                        offset = offset_col + offset_rows
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
