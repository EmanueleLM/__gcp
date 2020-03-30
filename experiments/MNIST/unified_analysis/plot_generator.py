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

from scipy import stats

import draw_bipartite as db


def regime(init, fin, mode, init_bias=None, fin_bias=None):        
    if mode == 'inh':
        init[init>=0.] = 0.
        fin[fin>=0.] = 0.
        init, fin = np.abs(init), np.abs(fin)
        if not isinstance(init_bias, type(None)):
            init_bias[init_bias>=0.] = 0.
            fin_bias[fin_bias>=0.] = 0.
            init_bias, fin_bias = np.abs(init_bias), np.abs(fin_bias)
            return init, fin, init_bias, fin_bias
        else:
            return init, fin
    elif mode == 'act':
        init[init<=0.] = 0.
        fin[fin<=0.] = 0.
        init, fin = np.abs(init), np.abs(fin)
        if not isinstance(init_bias, type(None)):
            init_bias[init_bias<=0.] = 0.
            fin_bias[fin_bias<=0.] = 0.
            return init, fin, init_bias, fin_bias
        else:
            return init, fin
    elif mode == 'abs':
        init, fin = np.abs(init), np.abs(fin)
        if not isinstance(init_bias, type(None)):
            init_bias, fin_bias = np.abs(init_bias), np.abs(fin_bias)
            return init, fin, init_bias, fin_bias
        else:
            return init, fin    
    elif mode == 'tpos':
        if not isinstance(init_bias, type(None)):
            min_i, min_f = np.abs(min(np.min(init), np.min(init_bias))), np.abs(min(np.min(fin), np.min(fin_bias)))
            init += min_i
            init_bias += min_i
            fin += min_f
            fin_bias += min_f
            return init, fin, init_bias, fin_bias
        else:
            min_i, min_f = np.abs(np.min(init)), np.abs(np.min(fin))
            init += min_i
            fin += min_f
            return init, fin
    elif mode == 'tneg':
        if not isinstance(init_bias, type(None)):
            min_i, min_f = np.abs(max(np.max(init), np.max(init_bias))), np.abs(max(np.max(fin), np.max(fin_bias)))
            init += min_i
            init_bias += min_i
            fin += min_f
            fin_bias += min_f
            init, fin = np.abs(init), np.abs(fin)
            init_bias, fin_bias = np.abs(init_bias), np.abs(fin_bias)
            return init, fin, init_bias, fin_bias
        else:
            min_i, min_f = np.abs(np.max(init)), np.abs(np.max(fin))
            init += min_i
            fin += min_f
            init, fin = np.abs(init), np.abs(fin)
            return init, fin
    elif mode == '':
        if not isinstance(init_bias, type(None)):
            return init, fin, init_bias, fin_bias
    return init, fin

################################################
###### Transition Phase #######
################################################
# Print transition phase with accuracy on x-axis
from colour import Color
red = Color("green")
colors = list(red.range_to(Color("red"),36))
save_couples = []
topology = 'cnn'
mode = 'tneg'
w_t_ranger = np.arange(0.1, 0.975, 0.025)
len_w_t_ranger = range(len(w_t_ranger))
colors = list(red.range_to(Color("red"),len(w_t_ranger)))
# Input Node strength
for acc, i in zip(w_t_ranger, len_w_t_ranger):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin = regime(init, fin, mode, init_bias=None, fin_bias=None)

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
plt.savefig('./images/{}/{}/{}-transition-node-strength-input-layer_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Output Node strength
for acc, i in zip(w_t_ranger, len_w_t_ranger):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)
    
    _, s_out_i = np.sum(init, axis=1), np.sum(init, axis=0) 
    _, s_out_f = np.sum(fin, axis=1), np.sum(fin, axis=0) 
    s_i = s_out_i.flatten() + init_bias.flatten()
    s_f = s_out_f.flatten() + fin_bias.flatten()
    plt.title('Transition Phase Node Strength Output Layer')
    plt.errorbar(acc, s_i.flatten().mean(), yerr=s_i.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, s_f.flatten().mean(), yerr=s_f.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.ylabel('Nodes Strength')
plt.savefig('./images/{}/{}/{}-transition-node-strength-output-layer_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Input Node disparity
for acc, i in zip(w_t_ranger, len_w_t_ranger):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)

    init, fin = regime(init, fin, mode, init_bias=None, fin_bias=None)
    
    s_in_i, s_in_f = np.sum(init, axis=1)**2, np.sum(fin, axis=1)**2
    Y_i = np.sum(init**2, axis=1)/s_in_i
    Y_f = np.sum(fin**2, axis=1)/s_in_f   
    Y_i = Y_i[~np.isnan(Y_i)]
    Y_f = Y_f[~np.isnan(Y_f)]
    plt.title('Nodes disparity Input Layer')
    plt.errorbar(acc, Y_i.flatten().mean(), yerr=Y_i.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, Y_f.flatten().mean(), yerr=Y_f.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.savefig('./images/{}/{}/{}-transition-node-disparity-input-layer_regime-{}.png'.format(topology, mode, topology, mode))
plt.xlabel('Accuracy')
plt.ylabel('Node Disparity')
plt.show()

# Output Node disparity
for acc, i in zip(w_t_ranger, len_w_t_ranger):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    
    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)
    
    s_in_i = (np.sum((init), axis=0) + init_bias)**2
    s_in_f = (np.sum((fin), axis=0) + fin_bias)**2
    init, fin, init_bias, fin_bias = init**2, fin**2, init_bias**2, fin_bias**2
    Y_i = (np.sum(init, axis=0)+init_bias)/s_in_i
    Y_f = (np.sum(fin, axis=0)+fin_bias)/s_in_f 
    Y_i = Y_i[~np.isnan(Y_i)]
    Y_f = Y_f[~np.isnan(Y_f)]
    plt.title('Nodes disparity Output Layer')
    plt.errorbar(acc, Y_i.flatten().mean(), yerr=Y_i.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, Y_f.flatten().mean(), yerr=Y_f.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.savefig('./images/{}/{}/{}-transition-node-disparity-output-layer_regime-{}.png'.format(topology, mode, topology, mode))
plt.xlabel('Accuracy')
plt.ylabel('Node Disparity')
plt.show()

# Mean and Standard Deviation
for acc, i in zip(w_t_ranger, len_w_t_ranger):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)

    init = np.concatenate((init.flatten(), init_bias.flatten()))
    fin = np.concatenate((fin.flatten(), fin_bias.flatten()))
    plt.title('Mean')
    plt.errorbar(acc, init.mean(), yerr=init.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.errorbar(acc+0.025, fin.mean(), yerr=fin.flatten().std(), fmt='--o', color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.ylabel('Meand')
plt.savefig('./images/{}/{}/{}-transition-mean-std_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Standard Deviation
for acc, i in zip(w_t_ranger, len_w_t_ranger):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)

    init = np.concatenate((init.flatten(), init_bias.flatten()))
    fin = np.concatenate((fin.flatten(), fin_bias.flatten()))
    plt.title('Standard Deviation')
    plt.scatter(acc, init.std(), color=str(colors[i]), alpha=1.0, label="Y_i first acc {0:.2f}".format(acc))
    plt.scatter(acc+0.025, fin.std(), color=str(colors[i]), alpha=1.0, label="Y_i last acc {0:.2f}".format(acc))
    #plt.legend(loc='best')
plt.xlabel('Accuracy')
plt.ylabel('Standard Deviation')
plt.savefig('./images/{}/{}/{}-transition-std_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

################################################
###### Differential Distributions #######
################################################
step = 0.1
arange_accuracy = np.arange(0.1, 1., step)       
num_colors = len(arange_accuracy)
colors_cumulative = list(red.range_to(Color("red"),num_colors))
# Link weights
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)
    if mode == 'inh' or mode == 'act':
        init = init[init>0.]
        fin = fin[fin>0.]
        init_bias = init_bias[init_bias>0.]
        fin_bias = fin_bias[fin_bias>0.]

    init = np.concatenate((init.flatten(), init_bias.flatten()))
    fin = np.concatenate((fin.flatten(), fin_bias.flatten()))
    plt.title('Histogram Link Weights Range {0:.2f}-{1:.2f} (accuracy step {2:.3f})'.format(arange_accuracy[0], 0.05+arange_accuracy[-1], step))
    density_i = stats.kde.gaussian_kde(init.flatten())
    density_f = stats.kde.gaussian_kde(fin.flatten())
    x_i = np.arange(init.min(), init.max(), .001)
    x_f = np.arange(fin.min(), fin.max(), .001)
    plt.plot(x_i, density_i(x_i), alpha=.5, color=str(colors_cumulative[i]))
    plt.plot(x_f, density_f(x_f), alpha=.5, color=str(colors_cumulative[i]))
    #plt.legend(loc='best')
plt.xlabel('Weights')
plt.ylabel('Prob(w<W)')
plt.savefig('./images/{}/{}/{}-histogram-link-weights_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()
    
# Node Strenght Input
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin = regime(init, fin, mode, init_bias=None, fin_bias=None)

    s_in_i, _ = np.sum(init, axis=1), np.sum(init, axis=0) 
    s_in_f, _ = np.sum(fin, axis=1), np.sum(fin, axis=0)
    s_i = s_in_i.flatten()
    s_f = s_in_f.flatten()

    if mode == 'inh' or mode == 'act':
        s_i = s_i[s_i>0.]
        s_f = s_f[s_f>0.]
    
    plt.title('Histogram Node Strength Input Layer Range {0:.2f}-{1:.2f} (accuracy step {2:.3f})'.format(arange_accuracy[0], 0.05+arange_accuracy[-1], step))
    density_i = stats.kde.gaussian_kde(s_i.flatten())
    density_f = stats.kde.gaussian_kde(s_f.flatten())
    x_i = np.arange(s_i.min(), s_i.max(), .001)
    x_f = np.arange(s_f.min(), s_f.max(), .001)
    plt.plot(x_i, density_i(x_i), alpha=.5, color=str(colors_cumulative[i]))
    plt.plot(x_f, density_f(x_f), alpha=.5, color=str(colors_cumulative[i]))
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('D(s)')
plt.savefig('./images/{}/{}/{}-histogram-node-strenght-in_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Node Strenght Output
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)

    _, s_out_i = np.sum(init, axis=1), np.sum(init, axis=0) 
    _, s_out_f = np.sum(fin, axis=1), np.sum(fin, axis=0)
    s_i = s_out_i.flatten() + init_bias.flatten()
    s_f = s_out_f.flatten() + fin_bias.flatten()
    
    if mode == 'inh' or mode == 'act':
        s_i = s_i[s_i>0.]
        s_f = s_f[s_f>0.]
        
    plt.title('Histogram Node Strength Output Layer Range {0:.2f}-{1:.2f} (accuracy step {2:.3f})'.format(arange_accuracy[0], 0.05+arange_accuracy[-1], step))
    density_i = stats.kde.gaussian_kde(s_i.flatten())
    density_f = stats.kde.gaussian_kde(s_f.flatten())
    x_i = np.arange(s_i.min(), s_i.max(), .001)
    x_f = np.arange(s_f.min(), s_f.max(), .001)
    plt.plot(x_i, density_i(x_i), alpha=.5, color=str(colors_cumulative[i]))
    plt.plot(x_f, density_f(x_f), alpha=.5, color=str(colors_cumulative[i]))
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('D(s)')
plt.savefig('./images/{}/{}/{}-histogram-node-strenght-out_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Disparity Input
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin = regime(init, fin, mode, init_bias=None, fin_bias=None)
    
    s_in_i, s_in_f = np.sum(init, axis=1)**2, np.sum(fin, axis=1)**2
    Y_i = np.sum(init**2, axis=1)/s_in_i
    Y_f = np.sum(fin**2, axis=1)/s_in_f   
    Y_i = Y_i[~np.isnan(Y_i)]
    Y_f = Y_f[~np.isnan(Y_f)]
    plt.title('Nodes Disparity Input Layer')
    density_i = stats.kde.gaussian_kde(Y_i.flatten())
    density_f = stats.kde.gaussian_kde(Y_f.flatten())
    x_i = np.arange(Y_i.min(), Y_i.max(), .001)
    x_f = np.arange(Y_f.min(), Y_f.max(), .001)
    plt.plot(x_i, density_i(x_i), alpha=.5, color=str(colors_cumulative[i]))
    plt.plot(x_f, density_f(x_f), alpha=.5, color=str(colors_cumulative[i]))
    #plt.legend(loc='best')
plt.xlabel('Node Disparity')
plt.ylabel('D(Y)')
plt.savefig('./images/{}/{}/{}-histogram-node-disparity-in_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Disparity Output
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin = regime(init, fin, mode, init_bias=None, fin_bias=None)

    s_in_i = (np.sum((init), axis=0) + init_bias)**2
    s_in_f = (np.sum((fin), axis=0) + fin_bias)**2
    init, fin, init_bias, fin_bias = init**2, fin**2, init_bias**2, fin_bias**2
    Y_i = (np.sum(init, axis=0)+init_bias)/s_in_i
    Y_f = (np.sum(fin, axis=0)+fin_bias)/s_in_f     
    Y_i = Y_i[~np.isnan(Y_i)]
    Y_f = Y_f[~np.isnan(Y_f)]
    plt.title('Nodes Disparity Output Layer')
    density_i = stats.kde.gaussian_kde(Y_i.flatten())
    density_f = stats.kde.gaussian_kde(Y_f.flatten())
    x_i = np.arange(Y_i.min(), Y_i.max(), .001)
    x_f = np.arange(Y_f.min(), Y_f.max(), .001)
    plt.plot(x_i, density_i(x_i), alpha=.5, color=str(colors_cumulative[i]))
    plt.plot(x_f, density_f(x_f), alpha=.5, color=str(colors_cumulative[i]))
    #plt.legend(loc='best')
plt.xlabel('Node Disparity')
plt.ylabel('D(Y)')
plt.savefig('./images/{}/{}/{}-histogram-node-disparity-out_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()


################################################
###### Cumulative Distributions #######
################################################
step = 0.1
arange_accuracy = np.arange(0.1, 1.0, step)       
num_colors = len(arange_accuracy)
colors_cumulative = list(red.range_to(Color("red"),num_colors))
# Cumulative Link weights
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)
    if mode == 'inh' or mode == 'act':
        init = init[init>0.]
        fin = fin[fin>0.]
        init_bias = init_bias[init_bias>0.]
        fin_bias = fin_bias[fin_bias>0.]

    init = np.concatenate((init.flatten(), init_bias.flatten()))
    fin = np.concatenate((fin.flatten(), fin_bias.flatten()))
    plt.title('Cumulative Link Weights Range {0:.2f}-{1:.2f} (accuracy step {2:.3f})'.format(arange_accuracy[0], 0.05+arange_accuracy[-1], step))
    bins1, bins2 = np.histogram(init.flatten(), 51)[-1], np.histogram(fin.flatten(), 50)[-1]
    bins1[-1] = bins2[-1] = np.inf
    plt.hist(init.flatten(), bins=bins1, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(fin.flatten(), bins=bins2, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Weights')
plt.ylabel('Prob(w<W)')
plt.savefig('./images/{}/{}/{}-cumulative-link-weights_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()
    
# Cumulative Node Strenght Input
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin = regime(init, fin, mode, init_bias=None, fin_bias=None)

    s_in_i, _ = np.sum(init, axis=1), np.sum(init, axis=0) 
    s_in_f, _ = np.sum(fin, axis=1), np.sum(fin, axis=0)
    s_i = s_in_i.flatten()
    s_f = s_in_f.flatten()

    if mode == 'inh' or mode == 'act':
        s_i = s_i[s_i>0.]
        s_f = s_f[s_f>0.]
    
    plt.title('Cumulative Node Strength Input Layer Range {0:.2f}-{1:.2f} (accuracy step {2:.3f})'.format(arange_accuracy[0], 0.05+arange_accuracy[-1], step))
    bins1, bins2 = np.histogram(s_i.flatten(), 51)[-1], np.histogram(s_f.flatten(), 50)[-1]
    bins1[-1] = bins2[-1] = np.inf
    plt.hist(s_i.flatten(), bins=bins1, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(s_f.flatten(), bins=bins2, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('Prob(s<S)')
plt.savefig('./images/{}/{}/{}-cumulative-node-strenght-in_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Cumulative Node Strenght Output
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)

    _, s_out_i = np.sum(init, axis=1), np.sum(init, axis=0) 
    _, s_out_f = np.sum(fin, axis=1), np.sum(fin, axis=0)
    s_i = s_out_i.flatten() + init_bias.flatten()
    s_f = s_out_f.flatten() + fin_bias.flatten()
    
    if mode == 'inh' or mode == 'act':
        s_i = s_i[s_i>0.]
        s_f = s_f[s_f>0.]
        
    plt.title('Cumulative Node Strength Output Layer Range {0:.2f}-{1:.2f} (accuracy step {2:.3f})'.format(arange_accuracy[0], 0.05+arange_accuracy[-1], step))
    bins1, bins2 = np.histogram(s_i.flatten(), 51)[-1], np.histogram(s_f.flatten(), 50)[-1]
    bins1[-1] = bins2[-1] = np.inf
    plt.hist(s_i.flatten(), bins=bins1, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(s_f.flatten(), bins=bins2, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('Prob(s<S)')
plt.savefig('./images/{}/{}/{}-cumulative-node-strenght-out_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Cumulative Node Disparity Input
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin = regime(init, fin, mode, init_bias=None, fin_bias=None)

    s_in_i, s_in_f = np.sum(init, axis=1)**2, np.sum(fin, axis=1)**2
    Y_i = np.sum(init**2, axis=1)/s_in_i
    Y_f = np.sum(fin**2, axis=1)/s_in_f   
    Y_i = Y_i[~np.isnan(Y_i)]
    Y_f = Y_f[~np.isnan(Y_f)]
    plt.title('Nodes Disparity Input Layer')
    
    bins1, bins2 = np.histogram(Y_i.flatten(), 51)[-1], np.histogram(Y_f.flatten(), 50)[-1]
    bins1[-1] = bins2[-1] = np.inf
    plt.hist(Y_i.flatten(), bins=bins1, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(Y_f.flatten(), bins=bins2, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('Prob(s<S)')
plt.savefig('./images/{}/{}/{}-cumulative-node-disparity-in_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()

# Cumulative Node Disparity Output
for acc, i in zip(arange_accuracy, range(num_colors)):
    init_acc_le, fin_acc_ge = (np.around(acc,3), np.around(acc+0.025,3)), (np.around(acc+0.025,3), np.around(acc+0.05,3))
    init_prefix, fin_prefix = 'fin', 'fin'
    init = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin = np.load('./results/{}_weights_npy/{}_weights_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        
    init_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, init_prefix, init_acc_le[0], init_acc_le[1]), allow_pickle=True)
    fin_bias = np.load('./results/{}_weights_npy/{}_bias_acc-{}-{}.npy'.format(topology, fin_prefix, fin_acc_ge[0], fin_acc_ge[1]), allow_pickle=True)        

    init, fin, init_bias, fin_bias = regime(init, fin, mode, init_bias=init_bias, fin_bias=fin_bias)

    s_in_i = (np.sum((init), axis=0) + init_bias)**2
    s_in_f = (np.sum((fin), axis=0) + fin_bias)**2
    init, fin, init_bias, fin_bias = init**2, fin**2, init_bias**2, fin_bias**2
    Y_i = (np.sum(init, axis=0)+init_bias)/s_in_i
    Y_f = (np.sum(fin, axis=0)+fin_bias)/s_in_f     
    Y_i = Y_i[~np.isnan(Y_i)]
    Y_f = Y_f[~np.isnan(Y_f)]
    plt.title('Nodes Disparity Output Layer')

    bins1, bins2 = np.histogram(Y_i.flatten(), 51)[-1], np.histogram(Y_f.flatten(), 50)[-1]
    bins1[-1] = bins2[-1] = np.inf
    plt.hist(Y_i.flatten(), bins=bins1, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i first acc {0:.2f}".format(acc), cumulative=True, normed=True)
    plt.hist(Y_f.flatten(), bins=bins2, color=str(colors_cumulative[i]), histtype='step', alpha=1.0, label="Y_i last acc {0:.2f}".format(acc), cumulative=True, normed=True)
    #plt.legend(loc='best')
plt.xlabel('Node Strength')
plt.ylabel('Prob(s<S)')
plt.savefig('./images/{}/{}/{}-cumulative-node-disparity-out_regime-{}.png'.format(topology, mode, topology, mode))
plt.show()


################################################
###### Bipartite Graph #######
################################################
# draw the adjacency graph of last layer, trained and non-trained networks
# eventually cutoff the quantiles
# Negative weights
if mode == '':  # doesn't make any sense with any other mode
    init_prefix, fin_prefix = 'fin', 'fin'
    for a,b in ((0.9,1.), (0.95,1.), (0.97,1.), (0.99,1.)):
        min_percentile, max_percentile = a, b
        save1 = './images/{}/{}/{}-init-graph-negative-weights_quantiles({}-{})'.format(topology, mode, topology, min_percentile, max_percentile)
        save2 = './images/{}/{}/{}-final-graph-negative-weights_quantiles({}-{})'.format(topology, mode, topology, min_percentile, max_percentile)
        #  init
        init = np.load('./results/{}_weights_npy/{}_weights_acc-0.1-0.125.npy'.format(topology, init_prefix), allow_pickle=True)
        qts1, qts2 = np.quantile(init.flatten(), min_percentile), np.quantile(init.flatten(), max_percentile)
        init[init>0.] = 0.
        init = np.absolute(init)
        init[init<qts1] = 0.
        init[init>qts2] = 0.
        db.draw_bipartite_graph(init,
                                actual_size=(init.shape[0],init.shape[1]),
                                title='Connectivity graph last layer 0.1-0.125-accuracy - Negative Weights',
                                showfig=True,
                                savefig=save1) 
        #  fin
        fin = np.load('./results/{}_weights_npy/{}_weights_acc-0.975-1.0.npy'.format(topology, fin_prefix), allow_pickle=True)
        qts1, qts2 = np.quantile(fin.flatten(), min_percentile), np.quantile(fin.flatten(), max_percentile)
        fin[fin>0.] = 0.
        fin = np.absolute(fin)
        fin[init<qts1] = 0.
        fin[fin>qts2] = 0.
        db.draw_bipartite_graph(fin, 
                                actual_size=(init.shape[0],10),
                                title='Connectivity graph last layer 0.975-1.0-percentile - Negative Weights',
                                showfig=True,
                                savefig=save2)
        
        # Positive weights
        save1 = './images/{}/{}/{}-init-graph-positive-weights_quantiles({}-{})'.format(topology, mode, topology, min_percentile, max_percentile)
        save2 = './images/{}/{}/{}-final-graph-positive-weights_quantiles({}-{})'.format(topology, mode, topology, min_percentile, max_percentile)
        #  init
        init = np.load('./results/{}_weights_npy/{}_weights_acc-0.1-0.125.npy'.format(topology, init_prefix), allow_pickle=True)
        qts1, qts2 = np.quantile(init.flatten(), min_percentile), np.quantile(init.flatten(), max_percentile)
        init[init<0.] = 0.
        init = np.absolute(init)
        init[init<qts1] = 0.
        init[init>qts2] = 0.
        db.draw_bipartite_graph(init,
                                actual_size=(init.shape[0],init.shape[1]),
                                title='Connectivity graph last layer 0.1-0.125-accuracy - Positive Weights',
                                showfig=True,
                                savefig=save1) 
        #  fin
        fin = np.load('./results/{}_weights_npy/{}_weights_acc-0.975-1.0.npy'.format(topology, fin_prefix), allow_pickle=True)
        qts1, qts2 = np.quantile(fin.flatten(), min_percentile), np.quantile(fin.flatten(), max_percentile)
        fin[fin<0.] = 0.
        fin = np.absolute(fin)
        fin[init<qts1] = 0.
        fin[fin>qts2] = 0.
        db.draw_bipartite_graph(fin, 
                                actual_size=(init.shape[0],10),
                                title='Connectivity graph last layer 0.975-1.0-percentile - Positive Weights',
                                showfig=True,
                                savefig=save2)
