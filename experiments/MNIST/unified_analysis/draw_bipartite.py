# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:18:12 2020

@author: Emanuele
"""

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import matching

def draw_bipartite_graph(ls, actual_size, title='', showfig=True, savefig=''):
    g = nx.Graph()
    a=['a'+str(i) for i in range(actual_size[0])]
    b=['b'+str(j) for j in range(actual_size[1])]
    g.add_nodes_from(a,bipartite=0)
    g.add_nodes_from(b,bipartite=1)  
    for i in range(actual_size[0]):
        for j in range(actual_size[1]):
            if ls[i][j] != 0:
                g.add_edge(a[i], b[j])
    pos_a={}
    x=0.100
    const=0.100
    y=1.0
    for i in range(len(a)):
        pos_a[a[i]]=[x,y-i*const]    
    xb=0.500
    pos_b={}
    for i in range(len(b)):
        pos_b[b[i]]=[xb,y-i*const]    
    nx.draw_networkx_nodes(g,pos_a,nodelist=a,node_color='r',node_size=30,alpha=0.8)
    nx.draw_networkx_nodes(g,pos_b,nodelist=b,node_color='b',node_size=30,alpha=0.8)    
    # edges
    pos={}
    pos.update(pos_a)
    pos.update(pos_b)
    #nx.draw_networkx_edges(g,pos,edgelist=nx.edges(g),width=1,alpha=0.8,edge_color='g')
    nx.draw_networkx_labels(g,pos,font_size=10,font_family='sans-serif')
    m=matching.maximal_matching(g)
    nx.draw_networkx_edges(g,pos,edgelist=m,width=1,alpha=0.8,edge_color='k')    
    plt.title(title)
    if showfig is True:
        plt.show()
    if savefig != '':
        plt.savefig(savefig+'.png')
        plt.savefig(savefig+'.svg')
    
