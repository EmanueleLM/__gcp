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
    noise_stdev = 2e-3
    name_flat_file = 'results/experiments_150iters_popsize1000_var2e-3/flat.npy'
    name_adj_matrices = 'results/experiments_150iters_popsize1000_var2e-3/adj_'
    net_parameters = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]
    elite_seed = [239554898, 105465088, 218365829, 203396497, 217698719, 109346344, 97770307, 148322777, 40290812, 53317582, 37400546, 22010658, 175723653, 165222218, 73768585, 222361649, 122267290, 183229434, 110253602, 236404051, 84153860, 126893068, 26600584, 13761399, 219667882, 217646206, 241830177, 9876773, 529557, 31049115, 177676138, 69558821, 243714980, 122743854, 104370347, 60772675, 122493426, 211212441, 84340898, 171023574, 191988092, 6004240, 141975672, 182382331, 60169248, 48856296, 212556591, 125285908, 98653799, 112007566, 217452140, 5555922, 45744348, 73503130, 161775644, 220445681, 117347905, 167087389, 192827476, 136018156, 131109619, 167837843, 182240958, 8391516, 229586495, 213735879, 97850511, 183893057, 194165031, 214936839, 224240891, 154115694, 138837282, 248682662, 216393363, 68549987, 232222394, 1476246, 151179556, 23831083, 67522189, 37046337, 30331306, 30809804, 180069300, 109853152, 142626223, 133024700, 213505916, 123561113, 138092197, 82592552, 175310019, 103121523, 23245329, 29484858, 144073463, 104990896, 201160449, 150056184, 1083072, 70741010, 113226387, 152457657, 170490740, 243782293, 132168255, 246204618, 16971863, 162624757, 141616186, 231377171, 45423013, 216817895, 61660178, 244733443, 114529972, 107288080, 182990464, 17546645, 55757113, 85103364, 239033993, 160621849, 173390540, 91280594, 239992665, 218336514, 227637236, 129418130, 1701712, 214555227]
    
    # reconstruct flat weights
    flat = param.get_flat_parameters(elite_seed,
                                     parameters=num_parameters,
                                     seed=seed_noise,
                                     count=noise_size,
                                     noise_stdev=noise_stdev, 
                                     save_to_file=False, 
                                     dst=name_flat_file)
    
    # from flat to adjacency, each flat vector is processed indipendently
    len_elite = len(elite_seed)
    for i in range(len_elite):
        
        print("Processing matrix n°", i, " out of ", len_elite-1)
        dst_adj_matrices = name_adj_matrices + "gen_" +str(i) + '.npy'
        param.get_adjacency_matrices(net_parameters,
                                     flat[i],
                                     save_to_file=True, 
                                     dst=dst_adj_matrices)
        
    """
    # read list of seeds and write the complete tree to file
    src_seeds = 'seeds/seeds_experiments_224iters_popsize10_var2e-3.txt'
    dst_tree = 'results/experiments_224iters_popsize10_var2e-3/tree'
    evotree.draw_tree(src_seeds, dst_tree, diff_gen=True, draw_labels=True)
    """
    