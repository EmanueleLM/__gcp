# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:48:55 2019

@author: Emanuele
Last layer characterization.
"""
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../../metrics/')

import cumulative as cm
import metrics as mt
import draw_bipartite as db


def filename_gt(path, accuracy_threshold, prefix='f'):
    files_ = []
    for g in glob.glob(path + prefix + '*.npy'):
        if float('0.'+g.split('.npy')[0].split('0.')[-1])>accuracy_threshold:
            files_.append(g)
    return files_
        
if __name__=="__main__":
    topology = 'cnn'
    init, fin = np.zeros(shape=(128,10)), np.zeros(shape=(128,10)) 
    for i in filename_gt('results/{}/'.format(topology), 0., prefix='i'):
        tmp = np.load(i, allow_pickle=True)
        init += tmp[-1]+tmp[-2]  # matrix and bias
    for f in filename_gt('results/{}/'.format(topology), 0.9, prefix='f'):
        tmp = np.load(f, allow_pickle=True)
        fin += tmp[-1]+tmp[-2]  # matrix and bias
    # average the values
    init /= len(filename_gt('results/{}/'.format(topology), 0., prefix='i'))
    fin /= len(filename_gt('results/{}/'.format(topology), 0.9, prefix='f'))
    # nodes mean and variance
    plt.title('Weights last layer')
    plt.hist(init.flatten(), bins=50, color='red', alpha=0.5, label="first gen.", normed=True)
    plt.hist(fin.flatten(), bins=50, color='blue', alpha=0.5, label="last gen.", normed=True)
    plt.legend(loc='best')
    plt.savefig('./images/{}/{}-mean-variance.png'.format(topology, topology))
    plt.savefig('./images/{}/{}-mean-variance.svg'.format(topology, topology))
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
    plt.savefig('./images/{}/{}-s-in_s-out.png'.format(topology, topology))
    plt.savefig('./images/{}/{}-s-in_s-out.svg'.format(topology, topology))
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
    plt.savefig('./images/{}/{}-s.png'.format(topology, topology))
    plt.savefig('./images/{}/{}-s.svg'.format(topology, topology))
    plt.show() 
    # nodes disparity Y_i
    s_in_i, s_in_f = np.sum(np.abs(init), axis=1), np.sum(np.abs(fin), axis=1)
    Y_i = np.sum(init**2, axis=1)/s_in_i
    Y_f = np.sum(fin**2, axis=1)/s_in_f            
    plt.title('Nodes disparity last layer')
    plt.scatter(s_in_i, Y_i.flatten(), color='blue', alpha=0.5, label="Y_i first gen.")
    plt.scatter(s_in_f, Y_f.flatten(), color='green', alpha=0.5, label="Y_i last gen.")
    plt.legend(loc='best')
    plt.savefig('./images/{}/{}-nodes-disparity.png'.format(topology, topology))
    plt.savefig('./images/{}/{}-nodes-disparity.svg'.format(topology, topology))
    plt.show() 
    # cumulative link weights
    plt.title('Cumulative link weights last layer')
    tmp = cm.cumulative_distribution(init)
    tmp_f = cm.cumulative_distribution(fin)
    plt.hist(init.flatten(), bins=50, color='red', alpha=0.5, histtype='step', label="first gen.", cumulative=True, normed=True)
    plt.hist(fin.flatten(), bins=50, color='blue', alpha=0.5, histtype='step', label="last gen.", cumulative=True, normed=True)
    plt.legend(loc='best')
    plt.savefig('./images/{}/{}-cumulative-link-weights.png'.format(topology, topology))
    plt.savefig('./images/{}/{}-cumulative-link-weights.svg'.format(topology, topology))
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
    plt.savefig('./images/{}/{}-cumulative-node-strength-in.png'.format(topology, topology))
    plt.savefig('./images/{}/{}-cumulative-node-strength-in.svg'.format(topology, topology))
    plt.show()
    plt.title('Cumulative nodes strength output last layer')
    plt.hist(i_out, bins=50, color='red', alpha=0.5, histtype='step', cumulative=True, normed=True)
    plt.hist(f_out, bins=50, color='blue', alpha=0.5, histtype='step', cumulative=True, normed=True)
    plt.legend(loc='best')
    plt.savefig('./images/{}/{}-cumulative-node-strength-out.png'.format(topology, topology))
    plt.savefig('./images/{}/{}-cumulative-node-strength-out.svg'.format(topology, topology))
    plt.show()
    # draw the adjacency graph of the n-th percentile of the values
    n_perc = 95
    #  init
    init_adj, prc = init, np.percentile(init, n_perc)
    init_adj[init_adj<prc] = 0.
    init_adj = np.hstack((init, np.zeros(shape=(128,118))))
    db.draw_bipartite_graph(init_adj, 
                            actual_size=(128,10),
                            title='Connectivity graph last layer {}-percentile'.format(n_perc),
                            showfig=True,
                            savefig='./images/{}/{}-init-graph'.format(topology, topology))
    #  fin
    fin_adj, prc = fin, np.percentile(fin, n_perc)
    fin_adj[fin_adj<prc] = 0.
    fin_adj = np.hstack((fin, np.zeros(shape=(128,118))))
    db.draw_bipartite_graph(fin_adj, 
                            actual_size=(128,10),
                            title='Connectivity graph last layer {}-percentile'.format(n_perc),
                            showfig=True,
                            savefig='./images/{}/{}-final-graph'.format(topology, topology))