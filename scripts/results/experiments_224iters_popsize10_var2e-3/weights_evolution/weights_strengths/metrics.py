# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:32:42 2019

@author: Emanuele

Extract metrics from a neural network graph as weights strengths from adjacency 
matrices (initial vs. final), Q(w) vs. w etc.

With this file you can extract, plot and save:
    - Q(w) vs w
    - <Y_i>
    - <Y_i>(k) vs k
    - s_in, s_out
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


weights_name = ['input', 'layer1_conv1', 'layer2_conv2', 'layer3_dense1', 'output_dense2']
weights_shapes = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]

# Calculate, plot and save the weights strengths, i.e. s_in, s_out and their sum
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
    plt.xlabel('[VALUES]: s_out + s_in')
    plt.ylabel('[FREQUENCY]: s_out + s_in')
    plt.legend(loc='upper right')
    plt.savefig('hist_nodes_strengths_init_'+weights_name[o]+'.svg')
    plt.savefig('hist_nodes_strengths_init_'+weights_name[o]+'.png')
    plt.pause(0.05)
    plt.hist(fin_s[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation', normed=True)
    plt.title("[NODES STRENGTHS LAST GEN]: " + weights_name[o])
    plt.xlabel('[VALUES]: s_out + s_in')
    plt.ylabel('[FREQUENCY]: s_out + s_in')
    plt.legend(loc='upper right')
    plt.savefig('hist_nodes_strengths_fin_'+weights_name[o]+'.svg')
    plt.savefig('hist_nodes_strengths_fin_'+weights_name[o]+'.png')
    plt.pause(0.05)
    print("Distance (norm) between two vectors is ", np.linalg.norm(fin_s[i].flatten()-init_s[i].flatten()))
    o += 1
plt.show()

# Calculate, plot and save the average strength <s>(k)
# It is calculated as the sum of input strengths for nodes with degree equal to k,
#  for k \in [min_degree, max_degree]
# Calculate, plot and save the weights strengths, i.e. s_in, s_out and their sum
s_k_init, s_k_fin = {}, {}
s_k_init['l1'] = i_s.item().get('i-l1')
s_k_init['l2'] = i_s.item().get('i-l2') 
s_k_init['l3'] = i_s.item().get('i-l3') 
s_k_init['l4'] = i_s.item().get('i-l4')
s_k_fin['l1'] = f_s.item().get('i-l1') 
s_k_fin['l2'] = f_s.item().get('i-l2') 
s_k_fin['l3'] = f_s.item().get('i-l3') 
s_k_fin['l4'] = f_s.item().get('i-l4')

# calculate the nodes connections during the two convolutions
i1, i2 = np.ones(shape=(1, 84, 84, 4)), np.ones(shape=(1,21,21,16))
k1, k2 = np.ones(shape=(8,8,4,16)), np.ones(shape=(4,4,16,32))
s1, s2 = [1,4,4,1], [1,2,2,1]
conv1 = tf.nn.conv2d(i1, k1, strides=s1, padding="SAME")
conv2 = tf.nn.conv2d(i2, k2, strides=s2, padding="SAME" )
with tf.Session() as sess:
    first_connectivity = sess.run(conv1)
    second_connectivity = sess.run(conv2)

first_connectivity = np.pad(first_connectivity, ((0,0), (2,2), (2,2), (0,0)), 'edge')

# check if the numbers make sense
assert first_connectivity.shape == (1, 25, 25, 16), "Convolution for the first layer must match with connectivity shape"
assert second_connectivity.shape == (1, 11, 11, 32), "Convolution for the second given layer must match with connectivity shape"

# node degrees is the number of arcs (i.e. weights) that 'feed' a node
nodes_degrees = {'l1': first_connectivity.flatten(), 
                 'l2': second_connectivity.flatten(),
                 'l3': 3872,
                 'l4': 256}  # for each layer that admits s_input, assign it the k-degree
colors = {'l1': 'red', 'l2': 'orange', 'l3': 'green', 'l4': 'blue'}

for key in s_k_init.keys():
    s_k_init[key] /= nodes_degrees[key]
    s_k_fin[key] /= nodes_degrees[key]
    
# plot <s>(k) of each layer: init strengths
x_ax = {'l1': first_connectivity.flatten(), 
        'l2': second_connectivity.flatten(),
        'l3': np.array([3872 for _ in range(len(s_k_init['l3']))]),
        'l4': np.array([256 for _ in range(len(s_k_init['l4']))])}

for key in s_k_init.keys():
   plt.title('[<s>(k) vs k]: FIRST GENERATION')
   plt.scatter(x_ax[key], s_k_init[key], color=colors[key], label=key)
   plt.xscale('log')
   plt.legend(loc='upper right')
   plt.xlabel('k'); plt.ylabel('<s>(k)')
   #plt.ylim(-3., 3.)
   plt.savefig('s_k_vs_k_init.png')
   plt.savefig('s_k_vs_k_init.svg')
plt.show()

# plot <s>(k) of each layer: fin strengths
for key in s_k_init.keys():
   plt.title('[<s>(k) vs k]: LAST GENERATION')
   plt.scatter(x_ax[key], s_k_fin[key], color=colors[key], label=key)
   plt.xscale('log')
   plt.legend(loc='upper right')
   plt.xlabel('k'); plt.ylabel('<s>(k)')   
   #plt.ylim(-3., 3.)
   plt.savefig('s_k_vs_k_fin.png')
   plt.savefig('s_k_vs_k_fin.svg')
plt.show()

# Calculate, plot and save <Y_i>(k) vs k, and for each indegree k, compare it to the curve 1/k
i_s_squared = np.load('dict_init_squared_strengths.npy', allow_pickle=True)
f_s_squared = np.load('dict_fin_squared_strengths.npy', allow_pickle=True)

Yi_init = {}
Yi_fin = {}
Yi_init['i-l1'] = i_s_squared.item().get('i-l1') / (i_s.item().get('i-l1') + i_s.item().get('o-l1'))**2
Yi_init['i-l2'] = i_s_squared.item().get('i-l2') / (i_s.item().get('i-l2') + i_s.item().get('o-l2'))**2
Yi_init['i-l3'] = i_s_squared.item().get('i-l3') / (i_s.item().get('i-l3') + i_s.item().get('o-l3'))**2
Yi_init['i-l4'] = i_s_squared.item().get('i-l4') / i_s.item().get('i-l4')**2
Yi_fin['i-l1'] = f_s_squared.item().get('i-l1') / (f_s.item().get('i-l1') + f_s.item().get('o-l1'))**2
Yi_fin['i-l2'] = f_s_squared.item().get('i-l2') / (f_s.item().get('i-l2') + f_s.item().get('o-l2'))**2
Yi_fin['i-l3'] = f_s_squared.item().get('i-l3') / (f_s.item().get('i-l3') + f_s.item().get('o-l3'))**2
Yi_fin['i-l4'] = f_s_squared.item().get('i-l4') / f_s.item().get('i-l4')**2

nodes_degrees = {'i-l1': first_connectivity.flatten(), 
                 'i-l2': second_connectivity.flatten(),
                 'i-l3': np.array([3872 for _ in range(len(s_k_init['l3']))]),
                 'i-l4': np.array([256 for _ in range(len(s_k_init['l4']))])}  # for each layer that admits s_input, assign it the k-degree

Y_i_init_flatten, Y_i_fin_flatten = np.array([]), np.array([])
degrees = np.array([])
for key in Yi_init.keys():
    Y_i_init_flatten = np.append(Y_i_init_flatten, Yi_init[key].flatten())
    Y_i_fin_flatten = np.append(Y_i_fin_flatten, Yi_fin[key].flatten())
    degrees = np.append(degrees, nodes_degrees[key].flatten())

Y_k_init, Y_k_fin = {}, {}
for unique_k in np.sort(np.unique(degrees)):
    where_is_k = np.argwhere(degrees==unique_k)
    Y_k_init[str(unique_k)] = Y_i_init_flatten[where_is_k]
    Y_k_fin[str(unique_k)] = Y_i_fin_flatten[where_is_k]
    
# plot <Y>(k) vs k init
for key in Y_k_init.keys(): 
   x = np.array([int(float(key)) for _ in range(len(Y_k_init[key]))])   
   plt.title('[<Y>(k) vs k]: FIRST GENERATION')
   plt.scatter(x, Y_k_init[key])
   plt.legend(loc='upper right')
plt.show()

# plot <Y>(k) vs k fin
for key in Y_k_fin.keys(): 
   x = np.array([int(float(key)) for _ in range(len(Y_k_fin[key]))])   
   plt.title('[<Y>(k) vs k]: FIRST GENERATION')
   plt.scatter(x, Y_k_fin[key])
   plt.legend(loc='upper right')
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