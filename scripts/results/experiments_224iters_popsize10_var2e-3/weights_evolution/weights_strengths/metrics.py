# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:32:42 2019

@author: Emanuele

Extract metrics from a neural network graph as weights strengths from adjacency 
matrices (initial vs. final), Q(w) vs. w etc.
"""

import matplotlib.pyplot as plt
import numpy as np


weights_name = ['input', 'layer1_conv1', 'layer2_conv2', 'layer3_dense1', 'output_dense2']
weights_shapes = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]

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

o = 0
for i in init_s.keys():
    plt.title("[NODES STRENGTHS FIRST GEN]: " + weights_name[o])
    plt.hist(init_s[i].flatten(), bins=50, color='red', alpha=0.5, label='First Generation', normed=True)
    plt.xlabel('Parameters values')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('hist_nodes_strengths_init_'+weights_name[o]+'.svg')
    plt.pause(0.05)
    plt.hist(fin_s[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation', normed=True)
    plt.title("[NODES STRENGTHS LAST GEN]: " + weights_name[o])
    plt.xlabel('Parameters values')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('hist_nodes_strengths_fin_'+weights_name[o]+'.svg')
    plt.pause(0.05)
    print("Distance (norm) between two vectors is ", np.linalg.norm(fin_s[i].flatten()-init_s[i].flatten()))
    o += 1
plt.show()

# Q(w) vs w of the best 35 networs (initial gen. vs. final gen.)
weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']
weights_shapes = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]

# average the best 35 nets' parameters
init_weights = np.load('../IN_0-0.npy', allow_pickle=True)
fin_weights = np.load('../BN_0-last.npy', allow_pickle=True)

for i in range(1,35):
    tmp1 = np.load('../IN_'+str(i)+'-0.npy', allow_pickle=True)
    tmp2 = np.load('../BN_'+str(i)+'-last.npy', allow_pickle=True)
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

# (x,y)-plot Q(w) vs w, inital and final, for each layer
Qw_init = np.load('init_Q_w.npy', allow_pickle=True)
Qw_fin = np.load('fin_Q_w.npy', allow_pickle=True)

colors = ['red', 'green', 'blue', 'black', 'yellow', 'green', 'brown', 'purple']

# 1) initial
for i in range(8): 
   x, y = zip(*sorted(zip(init_weights[i].flatten(), Qw_init[i].flatten())))   
   plt.title('[Q(w) vs w]: FIRST GENERATION')
   plt.plot(x, y, color=colors[i], label=weights_name[i])
   plt.legend(loc='upper right')
   plt.savefig('Qw_vs_w_init.png')
plt.show()

# 2) final
for i in range(8): 
   x, y = zip(*sorted(zip(fin_weights[i].flatten(), Qw_fin[i].flatten())))   
   plt.title('[Q(w) vs w]: FINAL GENERATION')
   plt.plot(x, y, color=colors[i], label=weights_name[i])
   plt.legend(loc='upper right')
   plt.savefig('Qw_vs_w_fin.png')
plt.show()

# Q(w) initial/final vs. w initial/final, every layer is plotted separately
for i in range(8):
    plt.title('[Q(w) vs w]: layer {}'.format(weights_name[i]))
    x, y = zip(*sorted(zip(init_weights[i].flatten(), Qw_init[i].flatten())))   
    plt.plot(x, y, label='Q(w) vs w initial', color='red')
    x_, y_ = zip(*sorted(zip(fin_weights[i].flatten(), Qw_fin[i].flatten())))   
    plt.plot(x_, y_, label='Q(w) vs w final', color='blue')
    plt.legend(loc='upper right')
    plt.pause(0.05)
plt.show()