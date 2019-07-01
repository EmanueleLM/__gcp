# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 14:47:47 2019

@author: Emanuele
"""

import parameters as param
import evolutionary_tree as evotree

if __name__ == '__main__':
    
    seed_noise = 123
    num_parameters = 1008450
    noise_size = 250000000
    noise_stdev = 5e-3
    name_flat_file = 'results/flat.npy'
    name_adj_matrices = 'results/adj.npy'
    net_parameters = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]
    
    elite_seed = [239554898, 105465088, 218365829, 203396497, 217698719, 109346344, 97770307, 148322777, 40290812, 53317582, 190085332, 164773031, 171399027, 2630078, 176702601, 93686253, 76095667, 200382773, 69601312, 175911248, 207164713, 200274814, 76037612, 59256456, 85403614, 245682959, 188503021, 42693002, 182306255, 124802595, 68253527, 106656666, 113680198, 216151345, 145379360, 210265685, 11864995, 28246340, 160234225, 57807095, 98635395, 72515607, 154401302, 115105836, 122063070, 172836820, 78177743, 236678513, 206261478, 185651350, 59946015, 5703900, 77957013, 26444669, 73989014, 124905757, 117674443, 71553854, 59426794, 93793486, 246681090, 94980589, 222163592, 108435042, 28068164, 188779814, 128104917, 4224352, 221727609, 2385570, 164137500]
    
    # reconstruct flat weights
    flat = param.get_flat_parameters(elite_seed,
                                     parameters=num_parameters,
                                     seed=seed_noise,
                                     count=noise_size,
                                     noise_stdev=noise_stdev, 
                                     save_to_file=True, 
                                     dst=name_flat_file)
    
    # from flat to adjacency
    adj = param.get_adjacency_matrices(net_parameters,
                                       flat[0],
                                       save_to_file=True, 
                                       dst=name_adj_matrices)
    
    src_seeds = 'seeds/seeds.txt'
    dst_tree = 'results/tree'    
    # read list of seeds and write the complete tree to file
    evotree.draw_tree(src_seeds, dst_tree)