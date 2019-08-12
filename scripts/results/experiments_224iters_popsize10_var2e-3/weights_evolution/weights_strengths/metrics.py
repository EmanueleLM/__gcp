# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:32:42 2019

@author: Emanuele

Extract metrics from a neural network graph as weights strengths from adjacency 
matrices (initial vs. final), Q(w) vs. w etc.
"""

import matplotlib.pyplot as plt
import numpy as np


# weights strengths
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

# Q(w) vs w of the best 35 networs (initial gen. vs. final gen.)
weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']
weights_shapes = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]

# average the best 35 nets' parameters
init_weights = np.load('../analysis_adj_fin0.npy', allow_pickle=True)
fin_weights = np.load('../analysis_adj_fin0.npy', allow_pickle=True)

for i in range(1,35):
    tmp1 = np.load('../analysis_adj_init'+str(i)+'.npy', allow_pickle=True)
    tmp2 = np.load('../analysis_adj_fin'+str(i)+'.npy', allow_pickle=True)
    for j in range(8):
        init_weights[j] += tmp1[j]
        fin_weights[j] += tmp2[j]

# average along the n best nets
init_weights = np.array([tmp/35. for tmp in init_weights])
fin_weights = np.array([tmp/35. for tmp in fin_weights])
Qw_init, Qw_fin = [], []

for i in range(len(init_weights)):
    
    print("Processing vector ", i, " out of 8.")
    tmp1 = init_weights[i].flatten()
    tmp2 = fin_weights[i].flatten()
    len_w = len(tmp1)
    for j in range(len_w):
        Qw_init.append(len(tmp1[tmp1>tmp1[j]])/len_w)
        Qw_fin.append(len(tmp2[tmp2>tmp2[j]])/len_w)
        
Qw_init = np.asarray(Qw_init)
Qw_fin = np.asarray(Qw_fin)
    
np.save('init_Q_w.npy', Qw_init)
np.save('fin_Q_w.npy', Qw_fin)

# reshape Qw_init and Qw_fin to the nn layers'shapes
offset = 0
tmp1, tmp2 = [], []
for i in range(8):
    tmp1.append(Qw_init[offset:offset+np.prod(weights_shapes[i])].reshape(*weights_shapes[i]))
    tmp2.append(Qw_fin[offset:offset+np.prod(weights_shapes[i])].reshape(*weights_shapes[i]))
    offset += np.prod(weights_shapes[i])
    
Qw_init = np.asarray(tmp1)
Qw_fin = np.asarray(tmp2)

# scatterplot Q(w) vs w, inital and final, for each layer
# 1) initial
for i in range(8):
    
   print(init_weights[i].shape, Qw_init[i].shape)
   y_ax = np.array([i for i in range(len(Qw_init[i].flatten()))])
   plt.title('[HISTOGRAM LAYER {}]: Q(w) vs w'.format(weights_name[i]))
   plt.scatter(y_ax, init_weights[i].flatten(), color='red', alpha=0.5, label='w',)
   plt.scatter(y_ax, Qw_init[i].flatten(), color='blue', alpha=0.5, label='Q(w)')
   plt.legend(loc='upper right')
   plt.savefig('Qw_vs_w_init_'+weights_name[i]+'.png')
   plt.pause(0.05)
plt.show()

# 2) final
for i in range(8):
   print(init_weights[i].shape, Qw_init[i].shape)
   y_ax = np.array([i for i in range(len(init_weights[i].flatten()))])
   plt.title('[HISTOGRAM LAYER {}]: Q(w) vs w'.format(weights_name[i]))
   plt.scatter(y_ax, fin_weights[i].flatten(), color='red', alpha=0.5, label='w')
   plt.scatter(y_ax, Qw_fin[i].flatten(), color='blue', alpha=0.5, label='Q(w)')
   plt.legend(loc='upper right')
   plt.pause(0.05)
   plt.savefig('Qw_vs_w_fin_'+weights_name[i]+'.png')
plt.show()

# Q(w) initial vs. Q(w) final, histograms
for i in range(8):
    plt.title('[HISTOGRAM LAYER {}]: Q(w): initial vs. final'.format(weights_name[i]))
    plt.hist(Qw_init[i].flatten(), bins=50, color='red', alpha=0.5, label='First Generation')
    plt.hist(Qw_fin[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation')
    plt.xlabel('Parameters values')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    #plt.savefig('hist_'+weights_name[i]+'_Qw_init_vs_fin.png')
    plt.pause(0.05)
    print("Distance (norm) between two vectors is ", np.linalg.norm(Qw_init[i].flatten()-Qw_fin[i].flatten()))
plt.show()