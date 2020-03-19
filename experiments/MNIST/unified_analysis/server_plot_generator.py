# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:52:24 2020

@author: Emanuele

Use this code to scan the results/fc (or cnn) folder to extract metrics from all the
 raw weights, i.e., not averaged, and plot histograms of
- link weights
- node strength (input-output)

TODO: - merge with plot_generator.py
"""

import glob
import matplotlib.pyplot as plt
import numpy as np

from colour import Color

results_folders = ["0.1-0.125", "0.125-0.15", "0.15-0.175", "0.175-0.2",
                    "0.2-0.225", "0.225-0.25", "0.25-0.275", "0.275-0.3",
                    "0.3-0.325", "0.325-0.35", "0.35-0.375", "0.375-0.4",
                    "0.4-0.425","0.425-0.45", "0.45-0.475", "0.475-0.5",
                    "0.5-0.525", "0.525-0.55", "0.55-0.575", "0.575-0.6",
                    "0.6-0.625", "0.625-0.65", "0.65-0.675", "0.675-0.7",
                    "0.7-0.725", "0.725-0.75", "0.75-0.775", "0.775-0.8",
                    "0.8-0.825", "0.825-0.85", "0.85-0.875", "0.875-0.9",
                    "0.9-0.925", "0.925-0.95", "0.95-0.975", "0.975-1.0"]

topology = 'fc'  
step = 0.025
num_colors = len(results_folders)
red = Color("green")
colors = list(red.range_to(Color("red"),num_colors))
# HISTOGRAMS
# Link weights histogram
for acc, i in zip(results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)        
    plt.hist(np.concatenate((total_weights.flatten(), total_bias.flatten())),  color=str(colors[i]), histtype='step', bins=1000, alpha=.5, cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_weights-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()    
    
# Node strenght input layer
for acc, i in zip(results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_input_layer = total_weights.reshape(n_files, 32, 10).sum(axis=-1)
    plt.hist(s_input_layer.flatten(), bins=1000, alpha=.5, color=str(colors[i]), histtype='step', cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_node_strenght_input-layer-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()

# Node strenght output layer
for acc, i in zip(results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_output_layer = total_weights.reshape(n_files, 32, 10).sum(axis=1) + total_bias.reshape(n_files, 10,)
    plt.hist(s_output_layer.flatten(), bins=1000, alpha=.5, color=str(colors[i]), histtype='step', cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_node_strenght_output-layer-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()

# Node disparity input layer
for acc, i in zip(results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        min_w, min_b = np.min(w[-2]), np.min(w[-1])
        min_ = np.abs(min(min_b, min_w))
        total_weights = np.append(w[-2]+min_, total_weights)
        total_bias = np.append(w[-1]+min_, total_bias)
    s_input_layer = total_weights.reshape(n_files, 32, 10).sum(axis=-1)
    Y = np.sum(total_weights.reshape(n_files, 32, 10)**2, axis=-1)/s_input_layer**2
    plt.hist(Y.flatten(), bins=1000, alpha=.5, color=str(colors[i]), histtype='step', cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_node_disparity_input-layer-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()

# Node disparity output layer
for acc, i in zip(results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        min_w, min_b = np.min(w[-2]), np.min(w[-1])
        min_ = np.abs(min(min_b, min_w))
        total_weights = np.append(w[-2]+min_, total_weights)
        total_bias = np.append(w[-1]+min_, total_bias)
    s_output_layer = total_weights.reshape(n_files, 32, 10).sum(axis=1) + total_bias.reshape(n_files, 10,)
    Y = (np.sum(total_weights.reshape(n_files, 32, 10)**2, axis=1)+total_bias.reshape(n_files, 10,)**2)/s_output_layer**2
    plt.hist(Y.flatten(), bins=1000, alpha=.5, color=str(colors[i]), histtype='step', cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_node_disparity_output-layer-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()

# ERRORBARS:
# weights mean and variance
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
        bias += w[-1]
    weights /= n_files
    bias /= n_files  
    wb = np.concatenate((weights.flatten(), bias.flatten()))
    plt.errorbar(a, wb.mean(), yerr=wb.std(), color=str(colors[i]), alpha=1.)
plt.savefig('./images/{}/errorbar_total_weights-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()    

# node strenght input
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
    weights /= n_files
    s_input_layer = weights.sum(axis=1)
    plt.errorbar(a, s_input_layer.mean(), yerr=s_input_layer.std(), color=str(colors[i]), alpha=.5)
plt.savefig('./images/{}/errorbar_node-strenght-accuracy({}-{}-step-{})-input.png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()   

# node strenght output
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
        bias += w[-1]
    weights /= n_files
    bias /= n_files
    s_output_layer = weights.sum(axis=0) + bias
    plt.errorbar(a, s_output_layer.mean(), yerr=s_input_layer.std(), color=str(colors[i]), alpha=.5)
plt.savefig('./images/{}/errorbar_node-strenght-accuracy({}-{}-step-{})-output.png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()   

# node disparity input
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
    weights /= n_files
    weights += np.abs(np.min(weights))
    s_input_layer = weights.sum(axis=1)
    Y = np.sum(weights**2, axis=1)/s_input_layer**2
    plt.errorbar(a, Y.mean(), yerr=Y.std(), color=str(colors[i]), alpha=.5)
plt.savefig('./images/{}/errorbar_node-disparity-accuracy({}-{}-step-{})-input.png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()  

# node disparity output
for a, acc, i in zip(np.arange(0.125,1.25,0.025), results_folders, range(num_colors)):
    n_files = len(glob.glob("./results/{}/{}/*.npy".format(topology, acc)))
    weights, bias = np.zeros((32,10)), np.zeros(10,)
    for file_ in glob.glob("./results/{}/{}/*.npy".format(topology, acc)):
        w = np.load(file_, allow_pickle=True)
        weights += w[-2]
        bias += w[-1]
    weights /= n_files
    bias /= n_files
    min_ = np.abs(min(np.min(weights), np.min(bias)))
    weights += min_
    bias += min_
    s_output_layer = weights.sum(axis=0) + bias
    Y = (np.sum(weights**2, axis=0)+bias**2)/s_output_layer**2
    plt.errorbar(a, Y.mean(), yerr=Y.std(), color=str(colors[i]), alpha=.5)
plt.savefig('./images/{}/errorbar_node-disparity-accuracy({}-{}-step-{})-output.png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close() 
    

"""
# Multimodal fitting of wieghts
from sklearn import mixture

gmix = mixture.GaussianMixture(n_components = 3, covariance_type = "full")
fitted = gmix.fit(np.concatenate((total_weights, total_bias)).reshape(-1,1))
data=np.concatenate((total_weights.flatten(), total_bias.flatten()))
y,x,_=plt.hist(data,1000,alpha=.3,label='data')
x=(x[1:]+x[:-1])/2 # for len(x)==len(y)
def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)
def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)
params = [-0.34266558, 0.01597775**0.5, 580.,
          -0.02447527, 0.01349047**0.5, 700.,
          0.28088089, 0.01203812**0.5, 630.]
plt.plot(x,trimodal(x,*params),color='red',lw=3,label='model')
plt.legend()
"""
