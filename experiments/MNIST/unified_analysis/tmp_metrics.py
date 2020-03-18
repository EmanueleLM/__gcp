# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 22:52:24 2020

@author: Emanuele

Use this code inside a folder that contains bias and weights in .npy format to extract all
 of them and plot histograms of
- link weights
- node strength (input-output)

TODO: - automatic scan of each folder we want to compute the metrics
      - merge with plot_generator.py
      - save results
"""

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

total_weights = np.array([])
total_bias = np.array([])
for pth in Path.cwd().iterdir():
    w = np.load(pth, allow_pickle=True)
    total_weights = np.append(w[-2].flatten(), total_weights)
    total_bias = np.append(w[-1].flatten(), total_bias)
    
# Link weights histogram
plt.hist(np.concatenate((total_weights.flatten(), total_bias.flatten())), bins=1000, alpha=1.0, cumulative=False, normed=False)
plt.savefig('./../../../images/fc/histogram_total_weights-accuracy(0.1-0.125).png')
plt.show()
# Node strenght input layer
s_input_layer = total_weights.reshape(1001, 32, 10).sum(axis=-1)
plt.hist(s_input_layer.flatten(), bins=1000, alpha=1.0, cumulative=False, normed=False)
plt.savefig('./../../../images/fc/histogram_total_node_strenght_input-layer-accuracy(0.1-0.125).png')
plt.show()
# Node strenght output layer
s_output_layer = total_weights.reshape(1001, 32, 10).sum(axis=1) + total_bias.reshape(1001, 10,)
plt.hist(s_output_layer.flatten(), bins=1000, alpha=1.0, cumulative=False, normed=False)
plt.savefig('./../../../images/fc/histogram_total_node_strenght_output-layer-accuracy(0.1-0.125).png')
plt.show()

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
