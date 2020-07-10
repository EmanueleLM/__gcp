# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:52:24 2020

@author: Emanuele

Use this code to scan the results/fc (or cnn) folder to extract metrics from all the
 raw weights, i.e., not averaged, and plot histograms of
- link weights
- node strength (input-output)

"""

import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from colour import Color

results_folders = [ "0.475-0.5",
                    "0.5-0.525",
                    "0.525-0.55",
                    "0.55-0.575",
                    "0.575-0.6",
                    "0.6-0.625",
                    "0.625-0.65",
                    "0.675-0.7",
                    "0.725-0.75",
                    "0.75-0.775",
                    "0.775-0.8",
                    "0.8-0.825"
                    ]

topology = 'fc'  
min_, max_, step = 0.475, 0.825, 0.025
bins = 100
num_colors = len(results_folders)
red = Color("green")
colors = list(red.range_to(Color("red"),num_colors))
files_pattern = "./results/@topology@/@accuracy@/*fixed-topology_random-normal-0-1.00*.npy"  # wildcards for topology and accuracy
files_pattern = files_pattern.replace('@topology@', topology)
saved_images_path = "./images/{}/".format(topology)
neurons_per_layer = [144*64, 32, 2]

###############################################################################
# HISTOGRAMS
###############################################################################

# LINK WEIGHTS

# EMBEDDING: Link weights histogram (LAYER -1)
print("\n[logger]: Link weights histogram")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))
    print("[logger]: {} files in folder {}".format(n_files, files_))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[0][4:260,:].flatten(), total_weights)
    density = stats.kde.gaussian_kde(total_weights.flatten())
    x = np.arange(-1.5, 1.5, .1)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}EMB_histogram_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close() 

# INPUT: Link weights histogram (LAYER 0)
print("\n[logger]: Link weights histogram")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))
    print("[logger]: {} files in folder {}".format(n_files, files_))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        w = np.delete(w, 0)  # don't consider the embedding
        total_weights = np.append(w[0].flatten(), total_weights)
        total_bias = np.append(w[1].flatten(), total_bias) 
    density = stats.kde.gaussian_kde(np.concatenate((total_weights.flatten(), total_bias.flatten())))
    x = np.arange(-1.5, 1.5, .1)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}L0_histogram_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()  

# LAYER 2: Link weights histogram
print("\n[logger]: Link weights histogram")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))
    print("[logger]: {} files in folder {}".format(n_files, files_))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        w = np.delete(w, 0)  # don't consider the embedding
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias) 
    density = stats.kde.gaussian_kde(np.concatenate((total_weights.flatten(), total_bias.flatten())))
    x = np.arange(-1.5, 1.5, .01)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}L1_histogram_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close() 


# NODE STRENGTH
# Node strenght embedding
print("\n[logger]: Node strenght I-L0 layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[0][4:260,:].flatten(), total_weights)
    s_input_layer = 1 + total_weights.reshape(n_files, 256, 64).sum(axis=-1)  # +1 for the implicit weights in each input
    density = stats.kde.gaussian_kde(s_input_layer.flatten())
    x = np.arange(s_input_layer.min(), s_input_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}EMB_histogram_total_node_strenght_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node strenght input layer (I-L0)
print("\n[logger]: Node strenght I-L0 layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[1].flatten(), total_weights)
    s_input_layer = 1 + total_weights.reshape(n_files, neurons_per_layer[0], neurons_per_layer[1]).sum(axis=-1)  # +1 for the implicit weights in each input
    density = stats.kde.gaussian_kde(s_input_layer.flatten())
    x = np.arange(s_input_layer.min(), s_input_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}I-L0_histogram_total_node_strenght_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node strenght L0-L1
print("\n[logger]: Node strenght L0-L1 layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    total_weights_o = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[1].flatten(), total_weights)  # 28*28,64
        total_bias = np.append(w[2].flatten(), total_bias)  # 64
        total_weights_o = np.append(w[3].flatten(), total_weights_o)  # 64,32
    s_output_layer = total_weights.reshape(n_files, neurons_per_layer[0], neurons_per_layer[1]).sum(axis=1) + total_bias.reshape(n_files, neurons_per_layer[1],)
    s_output_layer += total_weights_o.reshape(n_files, neurons_per_layer[1], neurons_per_layer[2]).sum(axis=-1)
    density = stats.kde.gaussian_kde(s_output_layer.flatten())
    x = np.arange(s_output_layer.min(), s_output_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}L0-L1_histogram_total_node_strenght_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node strenght L1-O
print("\n[logger]: Node strenght L1-O layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    total_weights_o = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)  # 32,10
        total_bias = np.append(w[-1].flatten(), total_bias)  # 10
    s_output_layer = total_weights.reshape(n_files, neurons_per_layer[1], neurons_per_layer[2]).sum(axis=1) + total_bias.reshape(n_files, neurons_per_layer[2],)
    s_output_layer += 1  # implicit weights in output
    density = stats.kde.gaussian_kde(s_output_layer.flatten())
    x = np.arange(s_output_layer.min(), s_output_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}L1-O_histogram_total_node_strenght_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# STANDARD DEVIATION
# Node strenght input layer (I-L0)
print("\n[logger]: Node std EMB layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[0][4:260,:].flatten(), total_weights)
    s_input_layer = 1 + total_weights.reshape(n_files, 256, 64).sum(axis=-1)  # +1 for the implicit weights in each input
    density = stats.kde.gaussian_kde(s_input_layer.std(axis=-1).flatten())
    x = np.arange(s_input_layer.min(), s_input_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}EMB_histogram_total_node_std_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node strenght input layer (I-L0)
print("\n[logger]: Node std I-L0 layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[1].flatten(), total_weights)
    s_input_layer = 1 + total_weights.reshape(n_files, neurons_per_layer[0], neurons_per_layer[1]).sum(axis=-1)  # +1 for the implicit weights in each input
    density = stats.kde.gaussian_kde(s_input_layer.std(axis=-1).flatten())
    x = np.arange(s_input_layer.min(), s_input_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}I-L0_histogram_total_node_std_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node std L0-L1
print("\n[logger]: Node std L0-L1 layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    total_weights_o = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[1].flatten(), total_weights)  # 28*28,64
        total_bias = np.append(w[2].flatten(), total_bias)  # 64
        total_weights_o = np.append(w[3].flatten(), total_weights_o)  # 64,32
    s_output_layer = total_weights.reshape(n_files, neurons_per_layer[0], neurons_per_layer[1]).sum(axis=1) + total_bias.reshape(n_files, neurons_per_layer[1],)
    s_output_layer += total_weights_o.reshape(n_files, neurons_per_layer[1], neurons_per_layer[2]).sum(axis=-1)
    density = stats.kde.gaussian_kde(s_input_layer.std(axis=-1).flatten())
    x = np.arange(s_output_layer.min(), s_output_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}L0-L1_histogram_total_node_std_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# Node std L2-O
print("\n[logger]: Node std L2-O layer")
for acc, i in zip(results_folders, range(num_colors)):
    print("[logger]: Processing results {}".format(acc))    
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    total_weights, total_bias = np.array([]), np.array([])
    total_weights_o = np.array([])
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)  # 32,10
        total_bias = np.append(w[-1].flatten(), total_bias)  # 10
    s_output_layer = total_weights.reshape(n_files, neurons_per_layer[1], neurons_per_layer[2]).sum(axis=1) + total_bias.reshape(n_files, neurons_per_layer[2],)
    s_output_layer += 1  # implicit weights in output
    density = stats.kde.gaussian_kde(s_input_layer.std(axis=-1).flatten())
    x = np.arange(s_output_layer.min(), s_output_layer.max(), .001)
    plt.plot(x, density(x), alpha=.5, color=str(colors[i]))
plt.savefig('{}L1-O_histogram_total_node_std_accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()


###############################################################################
# ERRORBARS (SCATTER):
###############################################################################
# LINK WEIGHTS
# weights mean and variance L0
print("\n[logger]: Errorbar mean-variance")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights = np.zeros((n_files, 256, 64))
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[0][4:260,:])
    plt.errorbar(a, weights.mean(), yerr=weights.std(), fmt='--o', color=str(colors[i]), alpha=1.)
plt.savefig('{}EMB_errorbar_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()  

# weights mean and variance L0
print("\n[logger]: Errorbar mean-variance")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((n_files, neurons_per_layer[0],neurons_per_layer[1])), np.zeros(n_files, neurons_per_layer[1],)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[1])
        bias = np.append(bias, w[2])
    wb = np.concatenate((weights.flatten(), bias.flatten()))
    plt.errorbar(a, wb.mean(), yerr=wb.std(), fmt='--o', color=str(colors[i]), alpha=1.)
plt.savefig('{}L0_errorbar_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()    

# weights mean and variance L1
print("\n[logger]: Errorbar mean-variance")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((n_files, neurons_per_layer[1],neurons_per_layer[2])), np.zeros(n_files, neurons_per_layer[2],)
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[-2])
        bias = np.append(bias, w[-1])
    wb = np.concatenate((weights.flatten(), bias.flatten()))
    plt.errorbar(a, wb.mean(), yerr=wb.std(), fmt='--o', color=str(colors[i]), alpha=1.)
plt.savefig('{}L1_errorbar_total_weights-accuracy({}-{}-step-{}).png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()

# NODE STRENGTH
# node strenght EMB
print("\n[logger]: Errorbar node strength I-L0")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights = np.zeros((n_files, 256, 64))
    for file_ in glob.glob(files_):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[0][4:260,:])
    s_input_layer = weights.sum(axis=-1) + 1  # implicit input bias
    plt.errorbar(a, s_input_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}EMB_errorbar_node-strenght-accuracy({}-{}-step-{})-input.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# node strenght I-L0
print("\n[logger]: Errorbar node strength I-L0")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights = np.zeros((n_files, neurons_per_layer[0], neurons_per_layer[1]))
    for (j, file_) in enumerate(glob.glob(files_)):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[1])
    s_input_layer = weights.sum(axis=-1) + 1  # implicit input bias
    plt.errorbar(a, s_input_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}I-L0_errorbar_node-strenght-accuracy({}-{}-step-{})-input.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# node strenght output L0-L1
print("\n[logger]: Errorbar node strength L0-L1")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((n_files, neurons_per_layer[0],neurons_per_layer[1])), np.zeros((n_files, neurons_per_layer[1],))
    weights_o = np.zeros((n_files, neurons_per_layer[1],neurons_per_layer[2]))
    for (j, file_) in enumerate(glob.glob(files_)):
        w = np.load(file_, allow_pickle=True)
        weights_o = np.append(weights_o, w[3])
        weights = np.append(weights, w[1])
        bias = np.append(bias, w[2])
    s_output_layer = weights.sum(axis=1) + bias
    s_output_layer += weights_o.sum(axis=-1)
    plt.errorbar(a, s_output_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}L0-L1_errorbar_node-strenght-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# node strenght output L1-O
print("\n[logger]: Errorbar node strength L2-O")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((n_files, neurons_per_layer[1],neurons_per_layer[2])), np.zeros((n_files, neurons_per_layer[2],))
    for (j, file_) in enumerate(glob.glob(files_)):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[3])
        bias = np.append(bias, w[4])
    s_output_layer = weights.sum(axis=1) + bias
    s_output_layer += 1  # implicit output bias
    plt.errorbar(a, s_output_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}L1-O_errorbar_node-strenght-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close() 

# STANDARD DEVIATION
# node std EMB
print("\n[logger]: Errorbar node std I-L0")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights = np.zeros((n_files, 256, 64))
    for (j, file_) in enumerate(glob.glob(files_)):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[0][4:260,:])
    s_input_layer = weights.sum(axis=-1) + 1  # implicit input bias
    plt.errorbar(a, s_input_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}EMB_errorbar_node-std-accuracy({}-{}-step-{})-input.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# node std output I-L0
print("\n[logger]: Errorbar node std L0-L1")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights = np.zeros((n_files, neurons_per_layer[0],neurons_per_layer[1]))
    for (j, file_) in enumerate(glob.glob(files_)):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[1])
    s_output_layer = weights.sum(axis=-1)
    plt.errorbar(a, s_output_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}L0-L1_errorbar_node-std-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close()   

# node std output L0-L1
print("\n[logger]: Errorbar node std L1-L2")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((n_files, neurons_per_layer[0],neurons_per_layer[1])), np.zeros((n_files, neurons_per_layer[1],))
    weights_o = np.zeros((n_files, neurons_per_layer[1],neurons_per_layer[2]))
    for (j, file_) in enumerate(glob.glob(files_)):
        w = np.load(file_, allow_pickle=True)
        weights_o = np.append(weights_o, w[3])
        weights = np.append(weights, w[1])
        bias = np.append(bias, w[2])
    s_output_layer = weights.sum(axis=1) + bias
    s_output_layer += weights_o.sum(axis=-1)
    plt.errorbar(a, s_output_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}L1-L2_errorbar_node-std-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close() 

# node std output L1-O
print("\n[logger]: Errorbar node std L2-O")
for a, acc, i in zip(np.arange(min_, max_, step), results_folders, range(num_colors)):
    print("\t  Processing results {}".format(acc))
    files_ = files_pattern.replace('@accuracy@', acc)    
    n_files = len(glob.glob(files_))    
    weights, bias = np.zeros((n_files, neurons_per_layer[1],neurons_per_layer[2])), np.zeros((n_files, neurons_per_layer[2],))
    for (j, file_) in enumerate(glob.glob(files_)):
        w = np.load(file_, allow_pickle=True)
        weights = np.append(weights, w[3])
        bias = np.append(bias, w[4])
    s_output_layer = weights.sum(axis=1) + bias
    s_output_layer += 1  # implicit output bias
    plt.errorbar(a, s_output_layer.mean(), yerr=s_input_layer.std(), fmt='--o', color=str(colors[i]), alpha=.5)
plt.savefig('{}L2-O_errorbar_node-std-accuracy({}-{}-step-{})-output.png'.format(saved_images_path,0.1, 1.0, step))
plt.show()
plt.close() 
