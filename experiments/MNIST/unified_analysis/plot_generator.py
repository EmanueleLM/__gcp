# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:48:55 2019

@author: Emanuele
Last layer characterization.
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../../metrics/')

import cumulative as cm
import metrics as mt
import draw_bipartite as db

save_couples = []

for acc in np.arange(0.1, 1.0, 0.05):
    topology = 'fc'
    savefig = False
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)
    # nodes mean and variance
    plt.title('Weights last layer')
    plt.hist(init.flatten(), bins=50, color='red', alpha=0.5, label="first gen.", normed=True)
    plt.hist(fin.flatten(), bins=50, color='blue', alpha=0.5, label="last gen.", normed=True)
    plt.legend(loc='best')
    if savefig is True:
        plt.savefig('./images/{}/{}-mean-variance-{}-{}.png'.format(topology, topology, init_acc_le, fin_acc_ge))
        plt.savefig('./images/{}/svg/{}-mean-variance-{}-{}.svg'.format(topology, topology, init_acc_le, fin_acc_ge))
    plt.show()
    # nodes strength, s_in and s_out
    s_in_i, s_out_i = np.sum(init, axis=1), np.sum(init, axis=0)
    s_in_f, s_out_f = np.sum(fin, axis=1), np.sum(fin, axis=0)
    plt.title('In-Out Nodes strength last layer')
    plt.hist(s_in_i.flatten(), bins=50, color='red', alpha=0.5, label="s_in first gen.", normed=True)
    plt.hist(s_in_f.flatten(), bins=50, color='yellow', alpha=0.5, label="s_in last gen.",normed=True)
    plt.hist(s_out_i.flatten(), bins=50, color='blue', alpha=0.5, label="s_out first gen.",normed=True)
    plt.hist(s_out_f.flatten(), bins=50, color='green', alpha=0.5,label="s_out last gen.", normed=True)
    plt.legend(loc='best')
    if savefig is True:
        plt.savefig('./images/{}/{}-s-in_s-out-{}-{}.png'.format(topology, topology, init_acc_le, fin_acc_ge))
        plt.savefig('./images/{}/svg/{}-s-in_s-out-{}-{}.svg'.format(topology, topology, init_acc_le, fin_acc_ge))
    plt.show() 
    # nodes strength s = s_in + s_out
    prec_i, prec_f = 0., 0.
    s_in_i, s_out_i = np.sum(init, axis=1), np.sum(init, axis=0) 
    s_in_f, s_out_f = np.sum(fin, axis=1), np.sum(fin, axis=0) 
    s_i = s_in_i.flatten() + prec_i
    prec_i = s_out_i.flatten()
    s_f = s_in_f.flatten() + prec_f
    prec_f = s_out_f.flatten()
    plt.title('Nodes strength last layer')
    plt.hist(s_i.flatten(), bins=50, color='blue', alpha=0.5, label="s first gen.", normed=True)
    plt.hist(s_f.flatten(), bins=50, color='green', alpha=0.5, label="s last gen.", normed=True)
    plt.legend(loc='best')
    if savefig is True:
        plt.savefig('./images/{}/{}-s-{}-{}.png'.format(topology, topology, init_acc_le, fin_acc_ge))
        plt.savefig('./images/{}/svg/{}-s-{}-{}.svg'.format(topology, topology, init_acc_le, fin_acc_ge))
    plt.show() 
    # nodes disparity Y_i
    s_in_i, s_in_f = np.sum(np.abs(init), axis=1), np.sum(np.abs(fin), axis=1)
    Y_i = np.sum(init**2, axis=1)/s_in_i
    Y_f = np.sum(fin**2, axis=1)/s_in_f            
    plt.title('Nodes disparity last layer')
    plt.scatter(s_in_i, Y_i.flatten(), color='blue', alpha=0.5, label="Y_i first gen.")
    plt.scatter(s_in_f, Y_f.flatten(), color='green', alpha=0.5, label="Y_i last gen.")
    plt.legend(loc='best')
    if savefig is True:
        plt.savefig('./images/{}/{}-nodes-disparity-{}-{}.png'.format(topology, topology, init_acc_le, fin_acc_ge))
        plt.savefig('./images/{}/svg/{}-nodes-disparity-{}-{}.svg'.format(topology, topology, init_acc_le, fin_acc_ge))
    plt.show() 
    # cumulative link weights
    plt.title('Cumulative link weights last layer')
    plt.hist(init.flatten(), bins=50, color='red', alpha=0.5, histtype='step', label="first gen.", cumulative=True, normed=True)
    plt.hist(fin.flatten(), bins=50, color='blue', alpha=0.5, histtype='step', label="last gen.", cumulative=True, normed=True)
    plt.legend(loc='best')
    if savefig is True:
        plt.savefig('./images/{}/{}-cumulative-link-weights-{}-{}.png'.format(topology, topology, init_acc_le, fin_acc_ge))
        plt.savefig('./images/{}/svg/{}-cumulative-link-weights-{}-{}.svg'.format(topology, topology, init_acc_le, fin_acc_ge))
    plt.show()
    # cumulative nodes strength
    plt.title('Cumulative nodes input strength last layer')
    i_in, i_out = mt.nodes_strength(init)
    f_in, f_out = mt.nodes_strength(fin)
    i_in = cm.cumulative_distribution(i_in)
    i_out = cm.cumulative_distribution(i_out)
    f_in = cm.cumulative_distribution(f_in)
    f_out = cm.cumulative_distribution(f_out)            
    plt.hist(i_in, bins=50, color='red', alpha=0.5, histtype='step', label="first gen.", cumulative=True, normed=True)
    plt.hist(f_in, bins=50, color='blue', alpha=0.5, histtype='step', label="last gen.", cumulative=True, normed=True)
    plt.legend(loc='best')
    if savefig is True:
        plt.savefig('./images/{}/{}-cumulative-node-strength-in-{}-{}.png'.format(topology, topology, init_acc_le, fin_acc_ge))
        plt.savefig('./images/{}/svg/{}-cumulative-node-strength-in-{}-{}.svg'.format(topology, topology, init_acc_le, fin_acc_ge))
    plt.show()
    plt.title('Cumulative nodes strength output last layer')
    plt.hist(i_out, bins=50, color='red', alpha=0.5, histtype='step', cumulative=True, normed=True)
    plt.hist(f_out, bins=50, color='blue', alpha=0.5, histtype='step', cumulative=True, normed=True)
    plt.legend(loc='best')
    if savefig is True:
        plt.savefig('./images/{}/{}-cumulative-node-strength-out-{}-{}.png'.format(topology, topology, init_acc_le, fin_acc_ge))
        plt.savefig('./images/{}/svg/{}-cumulative-node-strength-out-{}-{}.svg'.format(topology, topology, init_acc_le, fin_acc_ge))
    plt.show()
    # draw the adjacency graph of the n-th percentile of the values
    n_perc = 60
    save1 = './images/{}/{}-init-graph-{}-{}'.format(topology, topology, init_acc_le, fin_acc_ge)
    save2 = './images/{}/{}-final-graph-{}-{}'.format(topology, topology, init_acc_le, fin_acc_ge)
    #  init
    init_adj, prc = init, np.percentile(init, n_perc)
    init_adj[init_adj<prc] = 0.
    init_adj = np.hstack((init, np.zeros(shape=(init.shape[0],init.shape[0]-10))))
    db.draw_bipartite_graph(init_adj, 
                            actual_size=(init.shape[0],10),
                            title='Connectivity graph last layer {}-percentile'.format(n_perc),
                            showfig=True,
                            savefig=save1)
    #  fin
    fin_adj, prc = fin, np.percentile(fin, n_perc)
    fin_adj[fin_adj<prc] = 0.
    fin_adj = np.hstack((fin, np.zeros(shape=(init.shape[0],init.shape[0]-10))))
    db.draw_bipartite_graph(fin_adj, 
                            actual_size=(init.shape[0],10),
                            title='Connectivity graph last layer {}-percentile'.format(n_perc),
                            showfig=True,
                            savefig=save2)
  
################################################
###### Transition Phase and Cumulatives #######
################################################
# Print transition phase with accuracy on x-axis
from colour import Color
red = Color("green")
colors = list(red.range_to(Color("red"),36))
# Input Node strength
for acc, i in zip(np.arange(0.1, 0.975, 0.025), range(36)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    s_in_i, _ = np.sum(init, axis=1), np.sum(init, axis=0) 
    s_in_f, _ = np.sum(fin, axis=1), np.sum(fin, axis=0) 
    s_i = s_in_i.flatten()
    prec_i = s_in_i.flatten()
    s_f = s_in_f.flatten()
    plt.title('Transition Phase Node Strength Input Layer')
    plt.errorbar(acc, s_i.flatten().mean(), yerr=s_i.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, s_f.flatten().mean(), yerr=s_f.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.ylabel('Nodes Strength')
plt.savefig('./images/{}/{}-transition-node-strength-in.png'.format(topology, topology))
plt.show()

# Output Node strength
for acc, i in zip(np.arange(0.1, 0.975, 0.025), range(36)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    _, s_out_i = np.sum(init, axis=1), np.sum(init, axis=0) 
    _, s_out_f = np.sum(fin, axis=1), np.sum(fin, axis=0) 
    s_i = s_out_i.flatten()
    s_f = s_out_f.flatten()
    plt.title('Transition Phase Node Strength Output Layer')
    plt.errorbar(acc, s_i.flatten().mean(), yerr=s_i.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, s_f.flatten().mean(), yerr=s_f.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.ylabel('Nodes Strength')
plt.savefig('./images/{}/{}-transition-node-strength-out.png'.format(topology, topology))
plt.show()

# Input Node disparity
for acc, i in zip(np.arange(0.1, 0.975, 0.025), range(36)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)
    # nodes disparity Y_i
    s_in_i, s_in_f = np.sum(np.abs(init), axis=1), np.sum(np.abs(fin), axis=1)
    Y_i = np.sum(init**2, axis=1)/s_in_i
    Y_f = np.sum(fin**2, axis=1)/s_in_f            
    plt.title('Nodes disparity Input Layer')
    plt.errorbar(acc, Y_i.flatten().mean(), yerr=Y_i.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, Y_f.flatten().mean(), yerr=Y_f.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.savefig('./images/{}/{}-transition-node-disparity-in.png'.format(topology, topology))
plt.xlabel('Accuracy')
plt.ylabel('Node Disparity')
plt.show()

# Output Node disparity
for acc, i in zip(np.arange(0.1, 0.975, 0.025), range(36)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)
    # nodes disparity Y_i
    s_in_i, s_in_f = np.sum(np.abs(init), axis=0), np.sum(np.abs(fin), axis=0)
    Y_i = np.sum(init**2, axis=0)/s_in_i
    Y_f = np.sum(fin**2, axis=0)/s_in_f            
    plt.title('Nodes disparity Output Layer')
    plt.errorbar(acc, Y_i.flatten().mean(), yerr=Y_i.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, Y_f.flatten().mean(), yerr=Y_f.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.savefig('./images/{}/{}-transition-node-disparity-out.png'.format(topology, topology))
plt.xlabel('Accuracy')
plt.ylabel('Node Disparity')
plt.show()

# Mean ans Standard Deviation
for acc, i in zip(np.arange(0.1, 0.975, 0.025), range(36)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    plt.title('Mean')
    plt.errorbar(acc, init.mean(), yerr=init.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, fin.mean(), yerr=fin.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.ylabel('Meand')
plt.savefig('./images/{}/{}-transition-mean-std.png'.format(topology, topology))
plt.show()

# Standard Deviation
for acc, i in zip(np.arange(0.1, 0.975, 0.025), range(36)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    plt.title('Standard Deviation')
    plt.scatter(acc, init.std(), color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.scatter(acc+0.025, fin.std(), color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.ylabel('Standard Deviation')
plt.savefig('./images/{}/{}-transition-std.png'.format(topology, topology))
plt.show()
            
# Cumulative Link weights
arange_accuracy = np.arange(0.7, 0.975, 0.025)
num_colors = len(arange_accuracy)
colors_cumulative = list(red.range_to(Color("red"),num_colors))
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    plt.title('Cumulative Link Weights Range {0:.2f}-{1:.2f}'.format(arange_accuracy[0], 0.05+arange_accuracy[-1]))
    plt.hist(init.flatten(), bins=50, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(fin.flatten(), bins=50, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Weights')
plt.ylabel('Prob(w<W)')
plt.savefig('./images/{}/{}-cumulative-link-weights.png'.format(topology, topology))
plt.show()
    
# Cumulative Node Strenght Input
arange_accuracy = np.arange(0.1, 0.975, 0.025)
num_colors = len(arange_accuracy)
colors_cumulative = list(red.range_to(Color("red"),num_colors))
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    s_in_i, _ = np.sum(init, axis=1), np.sum(init, axis=0) 
    s_in_f, _ = np.sum(fin, axis=1), np.sum(fin, axis=0)
    s_i = s_in_i.flatten()
    s_f = s_in_f.flatten()
    plt.title('Cumulative Node Strength Input Layer Range {0:.2f}-{1:.2f}'.format(arange_accuracy[0], 0.05+arange_accuracy[-1]))
    plt.hist(s_i.flatten(), bins=50, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(s_f.flatten(), bins=50, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('Prob(s<S)')
plt.savefig('./images/{}/{}-cumulative-node-strenght-in.png'.format(topology, topology))
plt.show()

# Cumulative Node Strenght Output
arange_accuracy = np.arange(0.1, 0.975, 0.025)
num_colors = len(arange_accuracy)
colors_cumulative = list(red.range_to(Color("red"),num_colors))
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    _, s_out_i = np.sum(init, axis=1), np.sum(init, axis=0) 
    _, s_out_f = np.sum(fin, axis=1), np.sum(fin, axis=0)
    s_i = s_out_i.flatten()
    s_f = s_out_f.flatten()
    plt.title('Cumulative Node Strength Output Layer Range {0:.2f}-{1:.2f}'.format(arange_accuracy[0], 0.05+arange_accuracy[-1]))
    plt.hist(s_i.flatten(), bins=50, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(s_f.flatten(), bins=50, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('Prob(s<S)')
plt.savefig('./images/{}/{}-cumulative-node-strenght-out.png'.format(topology, topology))
plt.show()

# draw the adjacency graph of last layer, trained and non-trained networks
save1 = './images/{}/{}-init-graph-{}-{}'.format(topology, topology, init_acc_le, fin_acc_ge)
save2 = './images/{}/{}-final-graph-{}-{}'.format(topology, topology, init_acc_le, fin_acc_ge)
#  init
init = np.load('./results/{}_weights_npy/{}_weights_acc-0.1-0.125.npy'.format(topology, init_prefix), allow_pickle=True)
db.draw_bipartite_graph(init, 
                        actual_size=(init.shape[0],init.shape[1]),
                        title='Connectivity graph last layer 0.1-0.125-accuracy',
                        showfig=True,
                        savefig=save1) 
#  fin
fin = np.load('./results/{}_weights_npy/{}_weights_acc-0.975-1.0.npy'.format(topology, fin_prefix), allow_pickle=True)
db.draw_bipartite_graph(fin, 
                        actual_size=(init.shape[0],10),
                        title='Connectivity graph last layer 0.975-1.0-percentile',
                        showfig=True,
                        savefig=save2)
