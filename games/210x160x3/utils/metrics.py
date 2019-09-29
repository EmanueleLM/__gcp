# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:32:42 2019

@author: Emanuele

Extract metrics from a neural network graph as weights strengths from adjacency 
matrices (initial vs. final), Q(w) vs. w etc.

With this file you can extract, plot and save:
    - Q(w) vs w
    - <Y_i>
    - <Y_i>(k) vs k
    - s_in, s_out
    - Pk vs k
It is also possible to normalize the weights before calculating the various metric by applying 
 some normalizations techinques as in module normalization.py inside each folder that starts with
 '__normalization..'.
"""


def nodes_strength(i_s, f_s, dst, show=True):
    """
        Plot nodes' strengths, i.e. s_in, s_out
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    weights_name = ['input', 'layer1_conv1', 'layer2_conv2', 'layer3_dense1', 'output_dense2']
       
    init_s = {}
    fin_s = {}
    init_s = {'l0': i_s.item().get('o-l0')}
    fin_s = {'l0': f_s.item().get('o-l0')}
    init_s['l1'] = i_s.item().get('i-l1') + i_s.item().get('o-l1')
    init_s['l2'] = i_s.item().get('i-l2') + i_s.item().get('o-l2')
    init_s['l3'] = i_s.item().get('i-l3') + i_s.item().get('o-l3')
    init_s['l4'] = i_s.item().get('i-l4')
    fin_s['l1'] = f_s.item().get('i-l1') + f_s.item().get('o-l1')
    fin_s['l2'] = f_s.item().get('i-l2') + f_s.item().get('o-l2')
    fin_s['l3'] = f_s.item().get('i-l3') + f_s.item().get('o-l3')
    fin_s['l4'] = f_s.item().get('i-l4')
    
    o = 0
    for i in init_s.keys():
        plt.title("[NODES STRENGTHS FIRST GEN]: " + weights_name[o])
        plt.hist(init_s[i].flatten(), bins=50, color='red', alpha=0.5, label='First Generation', normed=True)
        plt.xlabel('[VALUES]: s_out + s_in')
        plt.ylabel('[FREQUENCY]: s_out + s_in')
        plt.legend(loc='upper right')
        plt.savefig('hist_nodes_strengths_init_'+weights_name[o]+'.svg')
        plt.savefig('hist_nodes_strengths_init_'+weights_name[o]+'.png')
        plt.pause(0.05)
        plt.hist(fin_s[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation', normed=True)
        plt.title("[NODES STRENGTHS LAST GEN]: " + weights_name[o])
        plt.xlabel('[VALUES]: s_out + s_in')
        plt.ylabel('[FREQUENCY]: s_out + s_in')
        plt.legend(loc='upper right')
        plt.savefig(dst + 'hist_nodes_strengths_fin_'+weights_name[o]+'.svg')
        plt.savefig(dst + 'hist_nodes_strengths_fin_'+weights_name[o]+'.png')
        plt.pause(0.05)
        print("[CUSTOM-LOGGER]: Distance (norm) between two vectors is {}.".format(np.linalg.norm(fin_s[i].flatten()-init_s[i].flatten())))
        o += 1
    
    if show == True:
        plt.show()
    else:
        pass


def avg_strength(i_s, f_s, card_i_s, card_f_s, dst, show=True):
    """
        Calculate, plot and save the average strength <s>(k)
        It is calculated as the sum of input strengths for nodes with degree equal to k,
         for k \in [min_degree, max_degree]
        Calculate, plot and save the weights strengths, i.e. s_in, s_out and their sum
    """
    
    import matplotlib.pyplot as plt

    s_k_init, s_k_fin = {}, {}
    s_k_init['l1'] = i_s.item().get('i-l1') + i_s.item().get('o-l1')  
    s_k_init['l2'] = i_s.item().get('i-l2') + i_s.item().get('o-l2')
    s_k_init['l3'] = i_s.item().get('i-l3') + i_s.item().get('o-l3')
    s_k_init['l4'] = i_s.item().get('i-l4')
    s_k_fin['l1'] = f_s.item().get('i-l1')  + i_s.item().get('o-l1')
    s_k_fin['l2'] = f_s.item().get('i-l2')  + i_s.item().get('o-l2')
    s_k_fin['l3'] = f_s.item().get('i-l3')  + i_s.item().get('o-l3')
    s_k_fin['l4'] = f_s.item().get('i-l4')
        
    card_init, card_fin = {}, {}
    card_init['l1'] = card_i_s.item().get('i-l1').flatten() + card_i_s.item().get('o-l1').flatten() +1  # bias edge
    card_init['l2'] = card_i_s.item().get('i-l2').flatten() + card_i_s.item().get('o-l2').flatten() +1
    card_init['l3'] = card_i_s.item().get('i-l3').flatten() + card_i_s.item().get('o-l3').flatten()
    card_init['l4'] = card_i_s.item().get('i-l4').flatten() +1  # bias and output edge, but bias has already taken into account
    card_fin['l1'] = card_f_s.item().get('i-l1').flatten() + card_i_s.item().get('o-l1').flatten() +1  # bias edge
    card_fin['l2'] = card_f_s.item().get('i-l2').flatten() + card_i_s.item().get('o-l2').flatten() +1
    card_fin['l3'] = card_f_s.item().get('i-l3').flatten() + card_i_s.item().get('o-l3').flatten()
    card_fin['l4'] = card_f_s.item().get('i-l4').flatten() +1  # bias and output edge, but bias has already taken into account
    
    colors = {'l1': 'red', 'l2': 'orange', 'l3': 'green', 'l4': 'blue'}
    
    for key in s_k_init.keys():
        s_k_init[key] /= len(card_init[key])
        s_k_fin[key] /= len(card_fin[key])
        
    # plot <s>(k) of each layer: init strengths
    for key in s_k_init.keys():
       plt.title('[<s>(k) vs k]: FIRST GENERATION')
       plt.scatter(card_init[key], s_k_init[key], color=colors[key], label=key)
       plt.xscale('log')
       plt.legend(loc='best')
       plt.xlabel('k'); plt.ylabel('<s>(k)')
       plt.savefig(dst + 's_k_vs_k_init.png')
       plt.savefig(dst + 's_k_vs_k_init.svg')
    plt.show()
    
    # plot <s>(k) of each layer: fin strengths
    for key in s_k_init.keys():
       plt.title('[<s>(k) vs k]: LAST GENERATION')
       plt.scatter(card_fin[key], s_k_fin[key], color=colors[key], label=key)
       plt.xscale('log')
       plt.legend(loc='best')
       plt.xlabel('k'); plt.ylabel('<s>(k)')   
       plt.savefig(dst + 's_k_vs_k_fin.png')
       plt.savefig(dst + 's_k_vs_k_fin.svg')
    
    if show == True:
        plt.show()
    else:
        pass


def Yk(i_s, f_s, card_i_s, card_f_s, i_s_squared, f_s_squared, dst, show=True):
    """
        Calculate, plot and save <Y_i>(k) vs k, and for each indegree k 
        (compare it to the curve 1/k)
    """
    import matplotlib.pyplot as plt
    import numpy as np
       
    Yi_init = {}
    Yi_fin = {}
    Yi_init['i-l1'] = i_s_squared.item().get('i-l1')
    Yi_init['i-l1'] /= (i_s.item().get('i-l1') + i_s.item().get('o-l1'))**2
    Yi_init['i-l2'] = i_s_squared.item().get('i-l2') 
    Yi_init['i-l2'] /= (i_s.item().get('i-l2') + i_s.item().get('o-l2'))**2
    Yi_init['i-l3'] = i_s_squared.item().get('i-l3') 
    Yi_init['i-l3'] /= (i_s.item().get('i-l3') + i_s.item().get('o-l3'))**2
    Yi_init['i-l4'] = i_s_squared.item().get('i-l4')
    Yi_init['i-l4'] /= (i_s.item().get('i-l4') +1)**2  # output strength is 1
    
    Yi_fin['i-l1'] = f_s_squared.item().get('i-l1') 
    Yi_fin['i-l1'] /= (f_s.item().get('i-l1') + f_s.item().get('o-l1'))**2
    Yi_fin['i-l2'] = f_s_squared.item().get('i-l2')
    Yi_fin['i-l2'] /= (f_s.item().get('i-l2') + f_s.item().get('o-l2'))**2
    Yi_fin['i-l3'] = f_s_squared.item().get('i-l3') +1  # output strength is 1
    Yi_fin['i-l3'] /= (f_s.item().get('i-l3') + f_s.item().get('o-l3'))**2
    Yi_fin['i-l4'] = f_s_squared.item().get('i-l4')
    Yi_fin['i-l4'] /= (f_s.item().get('i-l4') +1)**2  # output strength is 1
    
    nodes_degrees = {}
    for key in card_init.keys():
        nodes_degrees[key] = card_init[key]
    
    Y_i_init_flatten, Y_i_fin_flatten = np.array([]), np.array([])
    degrees = np.array([])
    for key in Yi_init.keys():
        Y_i_init_flatten = np.append(Y_i_init_flatten, Yi_init[key].flatten())
        Y_i_fin_flatten = np.append(Y_i_fin_flatten, Yi_fin[key].flatten())
        degrees = np.append(degrees, nodes_degrees[key.split('-')[-1]].flatten())
    
    Y_k_init, Y_k_fin = {}, {}
    for unique_k in np.sort(np.unique(degrees)):
        where_is_k = np.argwhere(degrees==unique_k)
        Y_k_init[str(unique_k)] = np.average(Y_i_init_flatten[where_is_k])
        Y_k_fin[str(unique_k)] = np.average(Y_i_fin_flatten[where_is_k])
        
    # plot <Y>(k) vs k init
    cols = ['red', 'orange', 'green', 'blue', 'black', 'grey']
    for (key, c) in zip(Y_k_init.keys(), cols): 
       x_ax = float(key)   
       plt.title('[<Y>(k) vs k]: FIRST GENERATION')
       plt.xscale('log')
       plt.scatter(x_ax, Y_k_init[key], color=c, label='k='+str(int(float(key))))
       plt.legend(loc='best')
       plt.savefig(dst + 'Y_k_vs_k_init.png')
       plt.savefig(dst + 'Y_k_vs_k_init.svg')
    if show == True:
        plt.show()
    else:
        pass
    
    # plot <Y>(k) vs k fin
    for (key, c) in zip(Y_k_init.keys(), cols): 
       x_ax = float(key)   
       plt.title('[<Y>(k) vs k]: LAST GENERATION')
       plt.xscale('log')
       plt.scatter(x_ax, Y_k_fin[key], color=c, label='k='+str(int(float(key))))
       plt.legend(loc='best')
       plt.savefig(dst + 'Y_k_vs_fin_samescale.png')
       plt.savefig(dst + 'Y_k_vs_k_fin_samescale.svg')  
       
    if show == True:
        plt.show()
    else:
        pass
  

def degrees_distribution(card, dst, show=True):
    """
        Plot the degree distribution of an undirected graph is defined as: pk = Nk/N
    """
    
    import matplotlib.pyplot as plt
    import numpy as np

    card_init = {}
    card_init['l1'] = card.item().get('i-l1').flatten() + card.item().get('o-l1').flatten() +1  # bias edge
    card_init['l2'] = card.item().get('i-l2').flatten() + card.item().get('o-l2').flatten() +1
    card_init['l3'] = card.item().get('i-l3').flatten() + card.item().get('o-l3').flatten()
    card_init['l4'] = card.item().get('i-l4').flatten() +1  # bias and output edge, but bias has already taken into account
    
    N = np.sum([Nk.shape[0] for Nk in card_init.values()])
    Pk, K, layer = list(), list(), list()
    # extract total number of links per node
    for key in card_init.keys():
        for k in np.unique(card_init[key]):
            K.append(k)
            Pk.append(len(np.argwhere(card_init[key]==k))/N)
            layer.append(key)
    
    cols = ['red', 'orange', 'green', 'blue', 'black', 'grey']
    label = [l + ': ' + str(k) for (l, k) in zip(layer, K)]
    fig, ax = plt.subplots()
    for (x, y, l, c) in zip(K, Pk, label, cols):
        plt.xscale('log')
        ax.scatter(x, y, c=c, label=l)
    
    plt.legend(loc='best')
    plt.title('[P_k vs k]: FIRST GENERATION')
    plt.xlabel('k')
    plt.ylabel('P(k)')
    plt.savefig('P_k_vs_init.png')
    plt.savefig('P_k_vs_k_init.svg')  
    
    if show == True:
        plt.show()
    else:
        pass    

def cumulative_link_weights(Qw_init, Qw_fin, init_weights, fin_weights, dst, show=True):
    """
        Plot Q(w) vs w, i.e the cumulative distribution of the link weights
    """
    
    import numpy as np
    import matplotli.pyplot as plt
    
    weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']    
    
    colors = ['red', 'green', 'blue', 'black', 'yellow', 'green', 'brown', 'purple']
    
    # 1) initial
    for i in range(8): 
       x, y = zip(*sorted(zip(init_weights[i].flatten(), Qw_init[i].flatten())))   
       plt.title('[Q(w) vs w]: FIRST GENERATION')
       plt.plot(x, y, color=colors[i], label=weights_name[i])
       plt.legend(loc='best')
       plt.savefig(dst + 'Qw_vs_w_init.png')

    if show == True:
        plt.show()
    else:
        pass 
    
    # 2) final
    for i in range(8): 
       x, y = zip(*sorted(zip(fin_weights[i].flatten(), Qw_fin[i].flatten())))   
       plt.title('[Q(w) vs w]: FINAL GENERATION')
       plt.plot(x, y, color=colors[i], label=weights_name[i])
       plt.legend(loc='best')
       plt.savefig(dst + 'Qw_vs_w_fin.png')

    if show == True:
        plt.show()
    else:
        pass 
    