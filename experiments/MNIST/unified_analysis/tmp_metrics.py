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

topology = 'fc'  
step = 0.2
arange_ = np.arange(0.175, 1.0, step)
num_colors = len(arange_)
red = Color("green")
colors = list(red.range_to(Color("red"),num_colors))
# Link weights histogram
for acc, i in zip(arange_, range(num_colors)):
    n_files = len(glob.glob("./results/{0}/{1:.3f}-{2:.1f}/*.npy".format(topology, acc, acc+0.025)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{0}/{1:.3f}-{2:.1f}/*.npy".format(topology, acc, acc+0.025)):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)        
    plt.hist(np.concatenate((total_weights.flatten(), total_bias.flatten())),  color=str(colors[i]), histtype='step', bins=1000, alpha=.5, cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_weights-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()    
    
# Node strenght input layer
for acc, i in zip(arange_, range(num_colors)):
    n_files = len(glob.glob("./results/{0}/{1:.3f}-{2:.1f}/*.npy".format(topology, acc, acc+0.025)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{0}/{1:.3f}-{2:.1f}/*.npy".format(topology, acc, acc+0.025)):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_input_layer = total_weights.reshape(n_files, 32, 10).sum(axis=-1)
    plt.hist(s_input_layer.flatten(), bins=1000, alpha=.5, color=str(colors[i]), histtype='step', cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_node_strenght_input-layer-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
plt.show()
plt.close()

# Node strenght output layer
for acc, i in zip(arange_, range(num_colors)):
    n_files = len(glob.glob("./results/{0}/{1:.3f}-{2:.1f}/*.npy".format(topology, acc, acc+0.025)))
    total_weights, total_bias = np.array([]), np.array([])
    for file_ in glob.glob("./results/{0}/{1:.3f}-{2:.1f}/*.npy".format(topology, acc, acc+0.025)):
        w = np.load(file_, allow_pickle=True)
        total_weights = np.append(w[-2].flatten(), total_weights)
        total_bias = np.append(w[-1].flatten(), total_bias)
    s_output_layer = total_weights.reshape(n_files, 32, 10).sum(axis=1) + total_bias.reshape(n_files, 10,)
    plt.hist(s_output_layer.flatten(), bins=1000, alpha=.5, color=str(colors[i]), histtype='step', cumulative=False, normed=False)
plt.savefig('./images/{}/histogram_total_node_strenght_output-layer-accuracy({}-{}-step-{}).png'.format(topology,0.1, 1.0, step))
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
