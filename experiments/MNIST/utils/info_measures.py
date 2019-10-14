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
        plt.title("Information Plane" + weights_name[o])
        plt.xlabel('I(X;T)')
        plt.ylabel('I(T;Y)')
        a, b = int(len(Ixt[i])/3), int(len(Ixt[i])*0.66)
        plt.scatter(Ixt[i][:a], Ity[i][:a], color='yellow', label='init')
        plt.scatter(Ixt[i][a:b], Ity[i][a:b], color='orange', label='mid')
        plt.scatter(Ixt[i][b:], Ity[i][b:], color='red', label='end')
        plt.legend(loc='best')
        plt.savefig(dst + 'information_plane_'+weights_name[o]+'.svg')
        plt.savefig(dst + 'information_plane_'+weights_name[o]+'.png')
        plt.pause(0.05)
        o += 1
    
    if show == True:
        plt.show()
    else:
        pass