# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:30:31 2019

@author: Emanuele

Plot Pareto front for a list of input seeds and episode's length.
"""

        
def pareto_frontier(src_scores, 
                    src_episodes_length, 
                    savefig_path=None, 
                    remove_outliers=None, 
                    maxX=True, 
                    maxY=True):
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    #Xs = np.genfromtxt(src_episodes_length) 
    #Ys = np.genfromtxt(src_scores)
    
    Xs = [1., 2., 3.]
    Ys = [4., 2., 4.]
    
    # remove_outliers removes pairs whose episode's length is higher than the input value
    if remove_outliers is not None:
        to_be_removed = np.argwhere(Xs >= remove_outliers)
        print("[CUSTOM-LOGGER]: elements removed due to remove_outliers == True: {}".format(to_be_removed))
        Xs = np.delete(Xs, to_be_removed)
        Ys = np.delete(Ys, to_be_removed)
        
    # Sort the list in either ascending or descending order of X
    myList = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
    
    # Start the Pareto frontier with the first value in the sorted list
    p_front = [myList[0]]    
    
    # Loop through the sorted list
    for pair in myList[1:]:
        if maxY: 
            if pair[1] >= p_front[-1][1]: # Look for higher values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
        else:
            if pair[1] <= p_front[-1][1]: # Look for lower values of Y…
                p_front.append(pair) # … and add them to the Pareto frontier
    
    # Turn resulting pairs back into a list of Xs and Ys
    p_frontX = [pair[0] for pair in p_front]
    p_frontY = [pair[1] for pair in p_front]
    
    plt.scatter(Xs, Ys, alpha=.5)        
    plt.plot(p_frontX, p_frontY, color='red', alpha=.75)
    plt.title('[PARETO FRONT]')
    plt.ylabel("Score")
    plt.xlabel("Episode length")
    plt.legend(['Pareto front'], loc='best')
    plt.savefig(savefig_path + 'pareto_front.png'); plt.savefig(savefig_path + 'pareto_front.svg')
    plt.show()
        