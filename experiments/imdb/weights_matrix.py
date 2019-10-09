# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 09:23:37 2019

@author: Emanuele
"""

import numpy as np


def get_weights_matrix(adj_matrices,
                       biases,
                       input_shapes,
                       transform=None):
    """
        Generate a dictionary with the weights matrices, given a series of adjacency matrices. For a formal
        definition of weights matrix, see def. 10.1, 10.2 chapter 10 Complex Networks
        by V. Latora et al. 
        It also generates a dictionary with the number of input arcs for each node (distinguishing between input and output). 
        adj_matrices:list of parameters, excluded biases;
        biases:list, list of biases, an entry can be None if bias for a specific layer is not present;
        input_shapes:list, list of input shapes i.e. the number of nodes in each layer;
        transform:function, specify a function that is used to process each weight before addin it up to the weights strengths.
                            It can be used to calculate metrics as Node disparity, def. 10.4 Chapter 10 Complex Networks book,
                            by specifying a lambda function like sq=lambda x: x**2 and passing it as argument to transform.
    """
       
    weights_strengths = {}
    nodes_cardinality = {}
       
    for i in range(len(adj_matrices)):
                                
        # output current layer
        weights_strengths['o-l'+str(i)] = np.sum(adj_matrices[i], axis=1).flatten()
        # input current layer
        weights_strengths['i-l'+str(i+1)] = np.sum(adj_matrices[i], axis=0).flatten()
        
        # take into account output biases
        if type(biases[i]) == np.ndarray:
            weights_strengths['i-l'+str(i+1)] += biases[i]                
            nodes_cardinality['o-l'+str(i)] = np.array([adj_matrices[i].shape[1] for _ in range(adj_matrices[i].shape[0])])
            nodes_cardinality['i-l'+str(i+1)] = np.array([adj_matrices[i].shape[0]+1 for _ in range(adj_matrices[i].shape[1])])
        
        else:                
            nodes_cardinality['o-l'+str(i)] = np.array([adj_matrices[i].shape[1] for _ in range(adj_matrices[i].shape[0])])
            nodes_cardinality['i-l'+str(i+1)] = np.array([adj_matrices[i].shape[0] for _ in range(adj_matrices[i].shape[1])])

                    
    return {'strengths': weights_strengths, 'cardinality': nodes_cardinality}

