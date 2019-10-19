# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:32:42 2019

@author: Emanuele

Mutual information, for each layer-to-layer.next()
"""


def plot_mutual_information(mutual_info, dst, show=True):
    
    import matplotlib.pyplot as plt
    
    weights_name = ['input-conv1', 'conv1-conv2', 'conv2-dense1', 'dense1-output']
       
    minfo = {}
    minfo = {'i-l1': mutual_info.item().get('i-l1')}
    minfo['l1-l2'] = mutual_info.item().get('l1-l2')
    minfo['l2-l3'] = mutual_info.item().get('l2-l3')
    minfo['l3-o'] = mutual_info.item().get('l3-o')
    
    o = 0
    for i in minfo.keys():
        plt.title("[MUTUAL INFO]: " + weights_name[o])
        plt.xlabel('')
        plt.ylabel('[MUTUAL INFO]')
        a, b = int(len(minfo[i])/3), int(len(minfo[i])*0.66)
        plt.scatter([t for t in range(len(minfo[i][:a]))], minfo[i][:a], color='yellow', label='init')
        plt.scatter([t for t in range(len(minfo[i][a:b]))], minfo[i][a:b], color='orange', label='mid')
        plt.scatter([t for t in range(len(minfo[i][b:]))], minfo[i][b:], color='red', label='end')
        plt.legend(loc='best')
        plt.savefig(dst + 'mutual_info_'+weights_name[o]+'.svg')
        plt.savefig(dst + 'mutual_info_'+weights_name[o]+'.png')
        plt.pause(0.05)
        o += 1
    
    if show == True:
        plt.show()
    else:
        pass
    
    
def plot_information_plane(I_x_t, I_t_y, dst, show=True):
    
    import matplotlib.pyplot as plt
    
    weights_name = ['input-conv1', 'conv1-conv2', 'conv2-dense1', 'dense1-output']
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    labels = ['1/5', '2/5', '3/5', '4/5', '5/5']
       
    Ixt, Ity = {}, {}
    Ixt = {'i-l1': I_x_t.item().get('i-l1')}
    Ixt['l1-l2'] = I_x_t.item().get('l1-l2')
    Ixt['l2-l3'] = I_x_t.item().get('l2-l3')
    Ixt['l3-o'] = I_x_t.item().get('l3-o')
    Ity = {'i-l1': I_t_y.item().get('i-o')}
    Ity['l1-l2'] = I_t_y.item().get('l1-o')
    Ity['l2-l3'] = I_t_y.item().get('l2-o')
    Ity['l3-o'] = I_t_y.item().get('l3-o')
    
    o = 0
    for i in Ixt.keys():
        X, T, Y = weights_name[o].split('-')[0], weights_name[o].split('-')[1], 'output'
        intervals = [j for j in range(0, len(Ixt[i])+int(len(Ixt[i])/5), int(len(Ixt[i])/5))]
        plt.title("Information Plane " + weights_name[o])
        plt.xlabel("I({};{})".format(X, T))
        plt.ylabel("I({};{})".format(T, Y))
        for j in range(len(intervals)-1):
            plt.scatter(Ixt[i][intervals[j]:intervals[j+1]], Ity[i][intervals[j]:intervals[j+1]], color=colors[j], label=labels[j])
        plt.legend(loc='best')
        plt.savefig(dst + 'information_plane_'+weights_name[o]+'.svg')
        plt.savefig(dst + 'information_plane_'+weights_name[o]+'.png')
        plt.pause(0.05)
        o += 1
    
    if show == True:
        plt.show()
    else:
        pass