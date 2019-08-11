# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:37:47 2019

@author: Emanuele

Generate histigrams of kernels and the respective dendograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

from sklearn.cluster import AgglomerativeClustering
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import dendrogram, linkage


init_weights = np.load('../analysis_adj_fin0.npy', allow_pickle=True)[0]
fin_weights = np.load('../analysis_adj_fin0.npy', allow_pickle=True)[0]

for i in range(1,35):
    tmp1 = np.load('../analysis_adj_init'+str(i)+'.npy', allow_pickle=True)[0]
    tmp2 = np.load('../analysis_adj_fin'+str(i)+'.npy', allow_pickle=True)[0]
    for j in range(8):
        init_weights[j] += tmp1[j]
        fin_weights[j] += tmp2[j]
        
# average along the n best nets
init_weights = np.array([tmp/35. for tmp in init_weights])
fin_weights = np.array([tmp/35. for tmp in fin_weights])

# first kernel extraction
init_rec_field1 = np.zeros(shape=(84, 84, 4, 16))
fin_rec_field1 = np.zeros(shape=(84, 84, 4, 16))

for i in range(16):
	for n in range(0, 80, 4):
		for m in range(0, 80, 4):
			init_rec_field1[n:n+8,m:m+8,:,i] += init_weights[:,:,:,i]
			fin_rec_field1[n:n+8,m:m+8,:,i] += fin_weights[:,:,:,i]

for i in range(16):
    plt.title('Initial receptive fields, n°' + str(i))
    plt.imshow(init_rec_field1[:,:,:3,i])  # first 3 channels as RGB
    plt.pause(0.05)
    plt.title('Final receptive fields, n°' + str(i))
    plt.imshow(fin_rec_field1[:,:,:3,i])  # first 3 channels as RGB
    plt.pause(0.05)
    print("Distance between receptive fields pre and post learning: ", np.linalg.norm(init_rec_field1[:,:,:3,i]-fin_rec_field1[:,:,:3,i]))
plt.show()

# generate clusters and dendograms with earth mover distance
dist = np.zeros(shape=(16,16))
for i in range(16):
    for j in range(16):
        dist[i,j] = wasserstein_distance(fin_rec_field1[:,:,:,i].flatten(), 
                                         fin_rec_field1[:,:,:,j].flatten())
        
cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage="average")
clusters = cluster.fit_predict(dist)

# plot dendogram
sq_dist = scp.spatial.distance.squareform(dist)
linkage_matrix = linkage(sq_dist, "single")
dendrogram(linkage_matrix)
plt.title("test")
plt.show()