# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:32:42 2019

@author: Emanuele

Extract weights strengths from adjacency matrices (initial vs. final)
"""

import matplotlib.pyplot as plt
import numpy as np


i_s = np.load('dict_init_strengths.npy', allow_pickle=True)
f_s = np.load('dict_fin_strengths.npy', allow_pickle=True)

init_s = {}
fin_s = {}
init_s = {'l0': i_s.item().get('o-l0')}
fin_s = {'l0': f_s.item().get('o-l0')}
init_s['l1'] = i_s.item().get('i-l1') + i_s.item().get('o-l1')
init_s['l2'] = i_s.item().get('i-l2') + i_s.item().get('o-l2')
init_s['l3'] = i_s.item().get('i-l3') + i_s.item().get('o-l3')
init_s['l4'] = i_s.item().get('i-l4')
fin_s['l1'] = f_s.item().get('i-l1') + f_s.item().get('o-l1')
fin_s['l2'] = f_s.item().get('i-l2') + f_s.item().get('o-l2')
fin_s['l3'] = f_s.item().get('i-l3') + f_s.item().get('o-l3')
fin_s['l4'] = f_s.item().get('i-l4')

for i in init_s.keys():
    plt.title(i)
    plt.hist(init_s[i].flatten(), bins=50, color='red', alpha=0.5, label='First Generation')
    plt.hist(fin_s[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation')
    plt.title(i)
    plt.xlabel('Parameters values')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.pause(0.05)
    print("Distance (norm) between two vectors is ", np.linalg.norm(fin_s[i].flatten()-init_s[i].flatten()))
plt.show()