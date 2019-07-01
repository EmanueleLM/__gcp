# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 15:13:16 2019

@author: Emanuele
"""

import networkx as nx
import matplotlib.pyplot as plt


def draw_tree(src, dst, draw_labels=True):    
    """
       src:string, must be in the format '/path/to/seeds_files/<name>', 
        where the file contains the lists of seeds of the element
        you want to draw the tree from. The file must be in form [list1]\n[list2]\n..
    """

    try:
        import pygraphviz
        from networkx.drawing.nx_agraph import graphviz_layout
    except ImportError:
        try:
            import pydot
            from networkx.drawing.nx_pydot import graphviz_layout
        except ImportError:
            raise ImportError("This example needs Graphviz and either "
                              "PyGraphviz or pydot")
    
    G = nx.DiGraph()
       
    with open(src) as fp:  
        
        line = fp.readline()
        cnt = 1
        
        while line:
            
            edges = list(line.replace(' ', '').strip("[]\n").split(','))
            graph_edges = []
            
            for i in range(len(edges)-1):
                
                connection = (edges[i], edges[i+1])
                graph_edges.append(connection)
                
            
            G.add_edges_from(graph_edges)
                
            line = fp.readline()
            cnt += 1
            
    #pos = nx.spring_layout(G)        
    pos = graphviz_layout(G, prog='twopi', args='')
    plt.figure(figsize=(100, 100))
    nx.draw(G, pos, node_size=500, alpha=0.5, node_color="blue", with_labels=draw_labels)
    plt.axis('equal')
    plt.savefig(dst+'.png', format='png')
    plt.savefig(dst+'.eps', format='eps')