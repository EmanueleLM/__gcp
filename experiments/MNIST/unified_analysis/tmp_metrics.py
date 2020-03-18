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
# Node strenght input layer
s_input_layer = total_weights.reshape(1001, 32, 10).sum(axis=-1)
plt.hist(s_input_layer.flatten(), bins=1000, alpha=1.0, cumulative=False, normed=False)
# Node strenght output layer
s_output_layer = total_weights.reshape(1001, 32, 10).sum(axis=1) + total_bias.reshape(1001, 10,)
plt.hist(s_output_layer.flatten(), bins=1000, alpha=1.0, cumulative=False, normed=False)
