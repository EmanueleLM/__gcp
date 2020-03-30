# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 20:37:47 2019

@author: Emanuele

Generate histigrams of kernels and the respective dendograms.
"""


def kernels(init_kernel, fin_kernel, dst, mode='greyscale', show=True):
    """
        Save (and plot) the kernels of the first layer, either in greyscale or RGB/RGBA format.
        Take as input the firts kernel, pre and post training.
    """
    
    import matplotlib.pyplot as plt
    
    if mode == 'greyscale':
        channels = 1
    elif mode == 'RGB':
        channels = 3
    else:
        channels = 4
    
    # plot each kernel as greyscale image
    for i in range(init_kernel.shape[-1]):
        max_, min_ = max(init_kernel.max(), fin_kernel.max()), min(init_kernel.min(), fin_kernel.min())  # share same min-max values for consistency
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title('[KERNEL FIRST GEN.]: Number n.{}'.format(str(i)))
        ax1.imshow(init_kernel[:,:,:channels,i].squeeze(), cmap='gray', vmin=min_, vmax=max_)  # first 3 channels as RGB, otherwise RGBA
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title('[KERNEL FIELD LAST GEN.]: Number n.{}'.format(str(i)))
        ax2.imshow(fin_kernel[:,:,:channels,i].squeeze(), cmap='gray', vmin=min_, vmax=max_)  # first 3 channels as RGB, otherwise RGBA
        plt.pause(0.05)
        fig.savefig(dst + 'kernels_conv1_' + mode +'_n{}.png'.format(str(i)))
    
    if show == True:
        plt.show()
    else:
        pass
    

def receptive_fields(init_kernel, fin_kernel, dst, mode='greyscale', show=True):
    """
        Save (and plot) the kernels of the first layer, either in greyscale or RGB/RGBA format.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    if mode == 'greyscale':
        channels = 1
    elif mode == 'RGB':
        channels = 3
    else:
        channels = 4
        
    # first rec. fields extraction
    init_rec_field1 = np.zeros(shape=(28, 28, channels, 16))
    fin_rec_field1 = np.zeros(shape=(28, 28, channels, 16))
    
    for i in range(init_rec_field1.shape[-1]):
        for n in range(0, 24, 4):
            for m in range(0, 24, 4):
                print(n, m)
                print(init_rec_field1[n:n+init_kernel.shape[0],m:m+init_kernel.shape[1],:channels,i].shape)
                init_rec_field1[n:n+init_kernel.shape[0],m:m+init_kernel.shape[1],:channels,i] += init_kernel[:,:,:channels,i]
                fin_rec_field1[n:n+init_kernel.shape[0],m:m+init_kernel.shape[1],:channels,i] += fin_kernel[:,:,:channels,i]
        
    for i in range(init_rec_field1.shape[-1]):
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax1.set_title('[RECEPTIVE FIELD FIRST GEN.]: Number n.{}'.format(str(i)))
        ax1.imshow(init_rec_field1[:,:,:channels,i].squeeze(), cmap='gray')  # first 3 channels as RGB
        ax2 = fig.add_subplot(2,2,2)
        ax2.set_title('[RECEPTIVE FIELD LAST GEN.]: Number n.{}'.format(str(i)))
        ax2.imshow(fin_rec_field1[:,:,:channels,i].squeeze(), cmap='gray')  # first 3 channels as RGB
        plt.pause(0.05)
        fig.savefig(dst + 'receptive_field_conv1_' + mode + '_n{}.png'.format(str(i)))
        print("[CUSTOM-LOGGER]: Distance between receptive fields pre and post learning: {}.".format(np.linalg.norm(init_rec_field1[:,:,:channels,i]-fin_rec_field1[:,:,:channels,i])))
    
    if show == True:
        plt.show()
    else:
        pass
    
    
def clustering(init_rec_field1, fin_rec_field1, dst, mode='RGB', show=True):
    """
        Dendograms with earth mover's distance.
    """    
    
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy as scp
    
    from sklearn.cluster import AgglomerativeClustering
    from scipy.stats import wasserstein_distance
    from scipy.cluster.hierarchy import dendrogram, linkage
    
    if mode == 'RGB':
        channels = 3
    else:
        channels = 4
    
    # first gen: generate clusters and dendograms with earth mover distance
    dist = np.zeros(shape=(16,16))
    for i in range(16):
        for j in range(16):
            dist[i,j] = wasserstein_distance(init_rec_field1[:,:,:channels,i].flatten(), 
                                             init_rec_field1[:,:,:channels,j].flatten())
            
    cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage="average")
    clusters = cluster.fit_predict(dist)
    
    # plot dendogram
    sq_dist = scp.spatial.distance.squareform(dist)
    linkage_matrix = linkage(sq_dist, "average")
    dendrogram(linkage_matrix)
    plt.title("[DENDOGRAM RECEPTIVE FIELDS, FIRST GENERATION]: earth mover distance, linkage 'avg'.")
    
    if show == True:
        plt.show()
    else:
        pass    
    # last gen: generate clusters and dendograms with earth mover distance
    dist = np.zeros(shape=(16,16))
    for i in range(16):
        for j in range(16):
            dist[i,j] = wasserstein_distance(fin_rec_field1[:,:,:channels,i].flatten(), 
                                             fin_rec_field1[:,:,:channels,j].flatten())
            
    cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage="average")
    clusters = cluster.fit_predict(dist)
    
    # plot dendogram
    sq_dist = scp.spatial.distance.squareform(dist)
    linkage_matrix = linkage(sq_dist, "average")
    dendrogram(linkage_matrix)
    plt.title("[DENDOGRAM RECEPTIVE FIELDS, LAST GENERATION]: earth mover distance, linkage 'avg'.")
    
    if show == True:
        plt.show()
    else:
        pass