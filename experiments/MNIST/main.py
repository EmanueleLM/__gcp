# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:47:47 2019

@author: Emanuele
"""
import json as json
import numpy as np
import sys as sys

import parameters as param
import weights_matrix as w_m

sys.path.append('./utils')
from cumulative_link_weights import Qw
from hist_mean_variance import hist_weights_mean_variance
from kernel_analysis import kernels, receptive_fields
from metrics import nodes_strength, avg_strength, Yk, degrees_distribution, cumulative_link_weights
from normalize import normalize_01


if __name__ == '__main__':
    
    # this is the only entry point you should touch in this file
    json_config = 'config/model.json'
    json_data = json.load(open(json_config))
    
    name_adj_matrices =  json_data['name_adj_matrices']
    
    # load initial and final weights, and normalize them
    init_weights = np.load(json_data['name_adj_matrices'] + 'init_params.npy', allow_pickle=True)
    fin_weights = np.load(json_data['name_adj_matrices'] + 'fin_params.npy', allow_pickle=True)    
    
    init_weights, fin_weights = normalize_01(init_weights, fin_weights)
    
    # standard parameters for a network whose input is an 84x84x4 image
    num_parameters = np.sum([np.prod(w.shape) for w in init_weights])
    net_parameters = [w.shape for w in init_weights]
                
    # save the plot of each network layer whose values are normalized between 0. and 1.
    print("\n[CUSTOM-LOGGER]: Extracting and saving weights mean and variance, for each layer.")
    hist_weights_mean_variance(init_weights, fin_weights, dst=json_data['dst_mean_variance'])

    # save the receptive fields and the kernels (RGB and/or RGBA) for the firts layer
    # TODO: second layer
    print("\n[CUSTOM-LOGGER]: Extracting and saving kernels and receptive fields (greyscale).")
    kernels(init_weights[0], fin_weights[0], dst=json_data['kernel_analysis'] + 'greyscale/', mode='greyscale', show=True)
    receptive_fields(init_weights[0], fin_weights[0], dst=json_data['kernel_analysis'] + 'receptive_fields/', mode='greyscale', show=True)
    
    
    # save the nodes strengths, cardinalities and squared node strengths
    print("\n[CUSTOM-LOGGER]: Extracting and saving node strengths, cardinalities and squared node strengths.")
    adj_matrices = [init_weights[0], init_weights[2], init_weights[4], init_weights[6]]
    biases = [init_weights[1], init_weights[3], init_weights[5], init_weights[7]]
    strides = [4,2, None, None]
    input_shapes = [(28,28,1), (21,21,16), (11,11,32), (3872), (256), (10)]
    input_padded = [(88,88,4), (25,25,16), (11,11,32), (3872), (256), (10)]
    init = w_m.get_weights_matrix(adj_matrices,
                                  biases,
                                  strides,
                                  input_shapes,
                                  input_padded,
                                  transform=None)
    
    adj_matrices = [fin_weights[0], fin_weights[2], fin_weights[4], fin_weights[6]]
    biases = [fin_weights[1], fin_weights[3], fin_weights[5], fin_weights[7]]
    fin = w_m.get_weights_matrix(adj_matrices,
                                 biases,
                                 strides,
                                 input_shapes,
                                 input_padded,
                                 transform=None)
    
    init_sq = w_m.get_weights_matrix(adj_matrices,
                                  biases,
                                  strides,
                                  input_shapes,
                                  input_padded,
                                  transform=np.square)
    
    fin_sq = w_m.get_weights_matrix(adj_matrices,
                                 biases,
                                 strides,
                                 input_shapes,
                                 input_padded,
                                 transform=np.square)
    
    print("[CUSTOM-LOGGER]: Saving node strengths, cardinalities and squared to folder {}.".format(json_data['metrics_path']))
    np.save(json_data['metrics_path'] + 'dict_fin_strengths.npy', fin['strengths'])
    np.save(json_data['metrics_path'] + 'dict_init_strengths.npy', init['strengths'])
    np.save(json_data['metrics_path'] + 'dict_init_cardinality.npy', init['cardinality'])
    np.save(json_data['metrics_path'] + 'dict_fin_cardinality.npy', fin['cardinality'])
    np.save(json_data['metrics_path'] + 'dict_fin_squared_strengths.npy', fin_sq['strengths'])
    np.save(json_data['metrics_path'] + 'dict_init_squared_strengths.npy', init_sq['strengths'])
    

    # Calculate, plot and save the weights strengths, i.e. s_in, s_out and their sum
    print("\n[CUSTOM-LOGGER]: Calculate, plot and save the weights strengths, i.e. s_in, s_out and their sum.")
    i_s = np.load(json_data['metrics_path'] + 'dict_init_strengths.npy', allow_pickle=True)
    f_s = np.load(json_data['metrics_path'] + 'dict_fin_strengths.npy', allow_pickle=True)    
    nodes_strength(i_s, f_s, dst=json_data['metrics_path'] + 's_in_s_out/', show=True)
    
    
    # Calculate, plot and save the average weights strengths, i.e. s_k vs k
    print("\n[CUSTOM-LOGGER]: Calculate, plot and save the average weights strengths, i.e. s_k vs k.")
    card_i_s = np.load(json_data['metrics_path'] + 'dict_init_cardinality.npy', allow_pickle=True)
    card_f_s = np.load(json_data['metrics_path'] + 'dict_fin_cardinality.npy', allow_pickle=True)
    avg_strength(i_s, f_s, card_i_s, card_f_s, dst=json_data['metrics_path'] + 's_k_vs_k/', show=True)
    
    
    # Calculate, plot and save <Y>(k) vs k metric
    print("\n[CUSTOM-LOGGER]: Calculate, plot and save <Y>(k) vs k metric.")
    i_s_squared = np.load(json_data['metrics_path'] + 'dict_init_squared_strengths.npy', allow_pickle=True)
    f_s_squared = np.load(json_data['metrics_path'] + 'dict_fin_squared_strengths.npy', allow_pickle=True)
    Yk(i_s, f_s, card_i_s, i_s_squared, f_s_squared, dst=json_data['metrics_path'] + 'Y_k_vs_k/', show=True)
    
    
    # Calculate, plot and save the degrees distribution Pk
    print("\n[CUSTOM-LOGGER]: Calculate, plot and save Pk vs k metric.")
    card = np.load(json_data['metrics_path'] + 'dict_init_cardinality.npy', allow_pickle=True)
    degrees_distribution(card, dst=json_data['metrics_path'] + 'Pk_vs_k/', show=True)
    

    # Calculate, plot and save the cumulative link weights
    print("\n[CUSTOM-LOGGER]: Calculate, plot and save Q(w) vs w.")
    print("[CUSTOM-LOGGER]: Please take care that this calculation may require hours!")
    Qw(init_weights, fin_weights, num_parameters, dst=json_data['metrics_path'])
    Qw_init = np.load(json_data['metrics_path'] + 'init_Q_w.npy', allow_pickle=True)
    Qw_fin = np.load(json_data['metrics_path'] + 'fin_Q_w.npy', allow_pickle=True)  
    cumulative_link_weights(init_weights, fin_weights, Qw_init, Qw_fin, dst=json_data['metrics_path'] + 'Qw_vs_w/')
    