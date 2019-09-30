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
        Generate a dictionary with the weights matrices, given a series of adjacency matrices. For a formal
        definition of weights matrix, see def. 10.1, 10.2 chapter 10 Complex Networks
        by V. Latora et al. 
        It also generates a dictionary with the number of input arcs for each node (distinguishing between input and output). 
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
    Speedup initial/final weights:
        adj_matrices = [fin_weights[0], fin_weights[2], fin_weights[4], fin_weights[6]]
        biases = [fin_weights[1], fin_weights[3], fin_weights[5], fin_weights[7]]
        strides = [4,2, None, None]
        input_shapes = [(84,84,4), (21,21,16), (11,11,32), (3872), (256), (18)]
        input_padded = [(88,88,4), (25,25,16), (11,11,32), (3872), (256), (18)]
        fin = get_weights_matrix(adj_matrices,
                               biases,
                               strides,
                               input_shapes,
                               input_padded,
                               transform=np.abs)  # you may want to set transform to np.abs
        # final
        adj_matrices = [init_weights[0], init_weights[2], init_weights[4], init_weights[6]]
        biases = [init_weights[1], init_weights[3], init_weights[5], init_weights[7]]
        strides = [4,2, None, None]
        input_shapes = [(84,84,4), (21,21,16), (11,11,32), (3872), (256), (18)]
        input_padded = [(88,88,4), (25,25,16), (11,11,32), (3872), (256), (18)]
        init = get_weights_matrix(adj_matrices,
                               biases,
                               strides,
                               input_shapes,
                               input_padded,
                               transform=np.abs)        
        np.save('dict_fin_strengths.npy', fin['strengths'])
        np.save('dict_init_strengths.npy', init['strengths'])
        np.save('dict_init_cardinality.npy', init['cardinality'])
        np.save('dict_fin_cardinality.npy', fin['cardinality'])
    """
    
    assert len(adj_matrices) == len(strides), "Each adj matrix should have an entry in the conv list (int or None)."
    len_params = len(adj_matrices)
    
    weights_strengths = {}
    nodes_cardinality = {}
       
    for i in range(len_params):
        
        # perform transformation to each weight before extracting weights
        if transform is not None:
            adj_matrices[i] = transform(adj_matrices[i])
            if len(biases) >= i:
                biases[i] = transform(biases[i])
               
        # first convolutional layer
        if i == 0:  # in the initial layers we need to consider also the input nodes
            
            # input strengths
            # reduce last dimension as it represents the number of filters
            current_ = np.sum(adj_matrices[i], axis=-1)  # 8x8x4x16 -> 8x8x4
            
            s = (input_shapes[1][0]*input_shapes[1][1],  # 21x21
                 input_padded[0][0]*input_padded[0][1],  # 88x88
                 input_shapes[0][2])  # 4 channels
            
            stride = strides[0]     
            one = input_shapes[1][2]*np.ones(shape=(current_.shape[0]))  # 16 filters in total for the input image
            weights_matrices = np.zeros(shape=s)
            nodes_connectivity = np.zeros(shape=s)
            
            for n in range(input_shapes[0][2]):
                for a in range(input_shapes[1][0]):
                    v_offset = a*stride*input_padded[0][0]  # vertical offset
                    for b in range(input_shapes[1][0]):
                        h_offset= b*stride  # horizontal offset
                        offset = v_offset + h_offset
                        for j in range(current_.shape[0]):
                            offset += (input_padded[0][0] if j!=0 else 0)  # offset for each entry in the 8x8 matrix
                            weights_matrices[(a*input_shapes[1][0]) + b, offset:offset+current_.shape[0], n] = current_[j,:,n]
                            nodes_connectivity[(a*input_shapes[1][0]) + b, offset:offset+current_.shape[0], n] += one

            weights_matrices = weights_matrices.reshape(input_shapes[1][0]*input_shapes[1][1],
                                                        input_padded[0][0],
                                                        input_padded[0][1],
                                                        input_shapes[0][2])
                                    
            nodes_connectivity = nodes_connectivity.reshape(input_shapes[1][0]*input_shapes[1][1],
                                                            input_padded[0][0],
                                                            input_padded[0][1],
                                                            input_shapes[0][2])
            
            # remove padding
            weights_matrices = weights_matrices[:,2:-2,2:-2,:]  # leave out 2 pixel per side: 88x88x4 -> 84x84x4
            weights_matrices = np.sum(weights_matrices, axis=0)
 
            nodes_connectivity = nodes_connectivity[:,2:-2,2:-2,:]  # leave out 2 pixel per side: 88x88x4 -> 84x84x4            
            nodes_connectivity = np.sum(nodes_connectivity, axis=0)
                        
            weights_strengths['o-l0'] = weights_matrices.flatten()
            nodes_cardinality['o-l0'] = nodes_connectivity.flatten()  
            

            # output strengths
            # reduce semi-last dimension as it represents the number of channels
            current_ = np.sum(adj_matrices[i], axis=-2)
            
            s = (input_shapes[1][0]*input_shapes[1][1],  # 21x21, output
                 input_padded[0][0]*input_padded[0][1],  # 88x88, input
                 input_shapes[1][2])  # 16 filters
            
            weights_matrices = np.zeros(shape=s)
            nodes_connectivity = np.zeros(shape=s)
            one = input_shapes[0][2]*np.ones(shape=(current_.shape[0]))  # !(note the multiplication): used to calculate node connectivity
            
            for n in range(s[2]):
                for a in range(input_shapes[1][0]):
                    v_offset = a*stride*input_padded[0][0]  # vertical offset
                    for b in range(input_shapes[1][0]):
                        h_offset= b*stride  # horizontal offset
                        offset = v_offset + h_offset
                        for j in range(current_.shape[0]):
                            offset += (input_padded[0][0] if j!=0 else 0)  # offset for each entry in the 8x8 matrix
                            weights_matrices[(a*input_shapes[1][0]) + b, offset:offset+current_.shape[0], n] = current_[j,:,n]
                            nodes_connectivity[(a*input_shapes[1][0]) + b, offset:offset+current_.shape[0], n] += one

            if len(biases) >= i:                
                weights_matrices = np.sum([weights_matrices, biases[i]], axis=-1)[0]
            
            weights_strengths['i-l1'] = np.sum(weights_matrices, axis=1).flatten()
            nodes_cardinality['i-l1'] = np.sum(nodes_connectivity, axis=1).reshape(input_shapes[1][0],input_shapes[1][1],input_shapes[1][2]).flatten() 
                
        # second convolutional layer
        if i == 1:
            
            # input strength
            # reduce last dimension as it represents the number of kernels
            current_ = np.sum(adj_matrices[i], axis=-1)
            
            s = (input_shapes[2][0]*input_shapes[2][1],  # 11x11, output
                 input_padded[1][0]*input_padded[1][1],  # 25x25, input
                 input_shapes[1][2])  # 16 channels
                                   
            stride = strides[1]            
            one = input_shapes[2][2]*np.ones(shape=(current_.shape[0]))  # used to calculate node connectivity
            weights_matrices = np.zeros(shape=s)
            nodes_connectivity = np.zeros(shape=s)
            
            for n in range(s[2]):
                for a in range(input_shapes[2][0]):
                    v_offset = a*stride*input_padded[1][0]  # vertical offset
                    for b in range(input_shapes[2][0]):
                        h_offset= b*stride  # horizontal offset
                        offset = v_offset + h_offset
                        for j in range(current_.shape[0]):
                            offset += (input_padded[1][0] if j!=0 else 0)  # offset for each entry in the 8x8 matrix
                            weights_matrices[(a*input_shapes[2][0]) + b, offset:offset+current_.shape[0], n] = current_[j,:,n]
                            nodes_connectivity[(a*input_shapes[2][0]) + b, offset:offset+current_.shape[0], n] += one
            
            weights_matrices = weights_matrices.reshape(input_shapes[2][0]*input_shapes[2][1],
                                                        input_padded[1][0],
                                                        input_padded[1][1],
                                                        input_shapes[1][2])
                                    
            nodes_connectivity = nodes_connectivity.reshape(input_shapes[2][0]*input_shapes[2][1],
                                                            input_padded[1][0],
                                                            input_padded[1][1],
                                                            input_shapes[1][2])
            
            # remove padding
            weights_matrices = weights_matrices[:,2:-2,2:-2,:]  # leave out 2 pixel per side: 25x25x16 -> 21x21x16
            weights_matrices = np.sum(weights_matrices, axis=0)
            
            nodes_connectivity = nodes_connectivity[:,2:-2,2:-2,:]  # leave out 2 pixel per side: 25x25x16 -> 21x21x16
            nodes_connectivity = np.sum(nodes_connectivity, axis=0)

            weights_strengths['o-l1'] = weights_matrices.flatten()
            nodes_cardinality['o-l1'] = nodes_connectivity.flatten() 
            
            # output strengths
            # reduce semi-last dimension as it represents the number of channels
            current_ = np.sum(adj_matrices[i], axis=-2)
            
            s = (input_shapes[2][0]*input_shapes[2][1],  # 11x11, output
                 input_padded[1][0]*input_padded[1][1],  # 25x25, input
                 input_shapes[2][2])  # 32 filters
            
            stride = strides[1]            
            one = input_shapes[1][2]*np.ones(shape=(current_.shape[0]))  # used to calculate node connectivity
            weights_matrices = np.zeros(shape=s)
            nodes_connectivity = np.zeros(shape=s)
            
            for n in range(s[2]):
                for a in range(input_shapes[2][0]):
                    v_offset = a*stride*input_padded[1][0]  # vertical offset
                    for b in range(input_shapes[2][0]):
                        h_offset= b*stride  # horizontal offset
                        offset = v_offset + h_offset
                        for j in range(current_.shape[0]):
                            offset += (input_padded[1][0] if j!=0 else 0)  # offset for each entry in the 8x8 matrix
                            weights_matrices[(a*input_shapes[2][0]) + b, offset:offset+current_.shape[0], n] = current_[j,:,n]
                            nodes_connectivity[(a*input_shapes[2][0]) + b, offset:offset+current_.shape[0], n] += one                        
            
            if len(biases) >= i:                
                weights_matrices = np.sum([weights_matrices, biases[i]], axis=-1)[0]

            # no need to remove padding
            weights_strengths['i-l2'] = np.sum(weights_matrices, axis=1).flatten()
            nodes_cardinality['i-l2'] = np.sum(nodes_connectivity, axis=1).reshape(input_shapes[2][0],input_shapes[2][1],input_shapes[2][2]).flatten()
            
        elif i == 2 or i == 3:  # dense layers are easy to manage
            
            # output current layer
            weights_strengths['o-l'+str(i)] = np.sum(adj_matrices[i], axis=1).flatten()
            # input current layer
            weights_strengths['i-l'+str(i+1)] = np.sum(adj_matrices[i], axis=0).flatten()
            
            # take into account output biases
            if len(biases) >= i:
                weights_strengths['i-l'+str(i+1)] += biases[i]                
                nodes_cardinality['o-l'+str(i)] = np.array([adj_matrices[i].shape[1] for _ in range(adj_matrices[i].shape[0])])
                nodes_cardinality['i-l'+str(i+1)] = np.array([adj_matrices[i].shape[0]+1 for _ in range(adj_matrices[i].shape[1])])
            
            else:                
                nodes_cardinality['o-l'+str(i)] = np.array([adj_matrices[i].shape[1] for _ in range(adj_matrices[i].shape[0])])
                nodes_cardinality['i-l'+str(i+1)] = np.array([adj_matrices[i].shape[0] for _ in range(adj_matrices[i].shape[1])])

                    
    return {'strengths': weights_strengths, 'cardinality': nodes_cardinality}

