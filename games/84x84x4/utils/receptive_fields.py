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

# isolate the initial kernel
init_weights = init_weights[0]
fin_weights = fin_weights[0]

# plot each kernel as RGBA image
for i in range(16):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('[KERNEL FIRST GEN.]: Number n.{}'.format(str(i)))
    ax1.imshow(init_weights[:,:,:3,i])  # first 3 channels as RGB
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('[KERNEL FIELD LAST GEN.]: Number n.{}'.format(str(i)))
    ax2.imshow(fin_weights[:,:,:3,i])  # first 3 channels as RGB
    plt.pause(0.05)
    fig.savefig('kernels_conv1_RGB_n{}.png'.format(str(i)))
plt.show()

# first rec. fields extraction
init_rec_field1 = np.zeros(shape=(84, 84, 4, 16))
fin_rec_field1 = np.zeros(shape=(84, 84, 4, 16))

for i in range(16):
	for n in range(0, 80, 4):
		for m in range(0, 80, 4):
			init_rec_field1[n:n+8,m:m+8,:,i] += init_weights[:,:,:,i]
			fin_rec_field1[n:n+8,m:m+8,:,i] += fin_weights[:,:,:,i]

for i in range(16):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('[RECEPTIVE FIELD FIRST GEN.]: Number n.{}'.format(str(i)))
    ax1.imshow(init_rec_field1[:,:,:,i])  # first 3 channels as RGB
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('[RECEPTIVE FIELD LAST GEN.]: Number n.{}'.format(str(i)))
    ax2.imshow(fin_rec_field1[:,:,:,i])  # first 3 channels as RGB
    plt.pause(0.05)
    fig.savefig('receptive_field_conv1_RGBA_n{}.png'.format(str(i)))
    print("Distance between receptive fields pre and post learning: ", np.linalg.norm(init_rec_field1[:,:,:3,i]-fin_rec_field1[:,:,:3,i]))
plt.show()

# first gen: generate clusters and dendograms with earth mover distance
dist = np.zeros(shape=(16,16))
for i in range(16):
    for j in range(16):
        dist[i,j] = wasserstein_distance(init_rec_field1[:,:,:,i].flatten(), 
                                         init_rec_field1[:,:,:,j].flatten())
        
cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage="average")
clusters = cluster.fit_predict(dist)

# plot dendogram
sq_dist = scp.spatial.distance.squareform(dist)
linkage_matrix = linkage(sq_dist, "average")
dendrogram(linkage_matrix)
plt.title("[DENDOGRAM RECEPTIVE FIELDS, FIRST GENERATION]: earth mover distance, linkage 'avg'.")
plt.show()

# last gen: generate clusters and dendograms with earth mover distance
dist = np.zeros(shape=(16,16))
for i in range(16):
    for j in range(16):
        dist[i,j] = wasserstein_distance(fin_rec_field1[:,:,:,i].flatten(), 
                                         fin_rec_field1[:,:,:,j].flatten())
        
cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage="average")
clusters = cluster.fit_predict(dist)

# plot dendogram
sq_dist = scp.spatial.distance.squareform(dist)
linkage_matrix = linkage(sq_dist, "average")
dendrogram(linkage_matrix)
plt.title("[DENDOGRAM RECEPTIVE FIELDS, LAST GENERATION]: earth mover distance, linkage 'avg'.")
plt.show()