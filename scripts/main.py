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
    name_adj_matrices = 'results/adj_'
    net_parameters = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]
    elite_seed = [23352730, 176157657, 145969510, 246293864, 201862725, 24667785, 172634372, 35985113, 183524492, 163490334, 7073303, 141247805, 233161899, 191397334, 123944276, 40044401, 236813780, 88531102, 177657539, 146545433, 126600553, 7995633, 222744946, 36690781, 29500347, 109500390, 169613370, 239524769, 230245490, 75491121, 164816317, 32933622, 232870780, 43930599, 110625277]
    
    # reconstruct flat weights
    flat = param.get_flat_parameters(elite_seed,
                                     parameters=num_parameters,
                                     seed=seed_noise,
                                     count=noise_size,
                                     noise_stdev=noise_stdev, 
                                     save_to_file=True, 
                                     dst=name_flat_file)
    
    # from flat to adjacency, each flat vector is processed indipendently
    len_elite = len(elite_seed)
    for i in range(len_elite):
        
        print("Processing matrix n°", i, " out of ", len_elite)
        dst_adj_matrices = name_adj_matrices + "gen_" +str(i) + '.npy'
        param.get_adjacency_matrices(net_parameters,
                                     flat[i],
                                     save_to_file=True, 
                                     dst=dst_adj_matrices)
  
    # read list of seeds and write the complete tree to file
    src_seeds = 'seeds/seeds.txt'
    dst_tree = 'results/tree'
    evotree.draw_tree(src_seeds, dst_tree, diff_gen=True, draw_labels=True)
    