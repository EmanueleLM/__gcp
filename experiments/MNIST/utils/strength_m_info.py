# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:15:49 2019

@author: Emanuele
"""

def strength_minfo(weights_s_minfo, dst, show=True):
    """
        Plot nodes strengths' mutual information thorugh the epochs
    """
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    import sys
    sys.path.append('../')
    import mutual_info as m_info
        
    # I(x,t)
    I_x_t = {'i-l1': np.array([]), 'l1-l2': np.array([]), 
             'l2-l3': np.array([]), 'l3-o': np.array([])}
    # I(t,y)
    I_t_y = {'i-o': np.array([]), 'l1-o': np.array([]), 
             'l2-o': np.array([]), 'l3-o': np.array([])}
    
    mi_limit_sample = 100
    tf_shapes = {}
    tf_shapes['i'] = len(weights_s_minfo[0]['o-l0'])
    tf_shapes['l1'] = len(weights_s_minfo[0]['i-l1'])
    tf_shapes['l2'] = len(weights_s_minfo[0]['i-l2'])
    tf_shapes['l3'] = len(weights_s_minfo[0]['i-l3'])
    tf_shapes['o'] = len(weights_s_minfo[0]['i-l4'])
    
    layers_indices = {}
    for key in tf_shapes.keys():
        layers_indices[key] = np.random.randint(0, tf_shapes[key], size=mi_limit_sample)


    for i in range(len(weights_s_minfo)):
        
        i_s = weights_s_minfo[i]
       
        init_s = {}
        init_s['i'] = i_s['o-l0']
        init_s['l1'] = i_s['o-l1']
        init_s['l2'] = i_s['o-l2']
        init_s['l3'] = i_s['o-l3']
        init_s['o'] = np.ones(shape=len(i_s['i-l4']))
        
        for key in ['i', 'l1', 'l2', 'l3']:
            if key == 'i':
                ixt = m_info.mutual_information((np.take(init_s['i'], indices=layers_indices['i']).reshape(-1, 1),
                                                np.take(init_s['l1'], indices=layers_indices['l1']).reshape(-1, 1)))
                ity = m_info.mutual_information((np.take(init_s['l1'], indices=layers_indices['l1']).reshape(-1, 1),
                                                 np.take(init_s['o'], indices=layers_indices['o']).reshape(-1, 1)))
                kk = 'i-l1'
            elif key == 'l1':
                ixt = m_info.mutual_information((np.take(init_s['l1'], indices=layers_indices['l1']).reshape(-1, 1),
                                                 np.take(init_s['l2'], indices=layers_indices['l2']).reshape(-1, 1)))
                ity = m_info.mutual_information((np.take(init_s['l2'], indices=layers_indices['l2']).reshape(-1, 1),
                                                 np.take(init_s['o'], indices=layers_indices['o']).reshape(-1, 1)))
                kk = 'l1-l2'
            elif key == 'l2':
                ixt = m_info.mutual_information((np.take(init_s['l2'], indices=layers_indices['l2']).reshape(-1, 1),
                                                 np.take(init_s['l3'], indices=layers_indices['l3']).reshape(-1, 1)))
                ity = m_info.mutual_information((np.take(init_s['l3'], indices=layers_indices['l3']).reshape(-1, 1),
                                                 np.take(init_s['o'], indices=layers_indices['o']).reshape(-1, 1)))
                kk = 'l2-l3'
            elif key == 'l3':
                ixt = m_info.mutual_information((np.take(init_s['l3'], indices=layers_indices['l3']).reshape(-1, 1),
                                                 np.take(init_s['o'], indices=layers_indices['o']).reshape(-1, 1)))
                ity = m_info.mutual_information((np.take(init_s['l3'], indices=layers_indices['l3']).reshape(-1, 1),
                                                 np.take(init_s['o'], indices=layers_indices['o']).reshape(-1, 1)))
                kk = 'l3-o'
                
            # save partial values of I(x,t) and I(t,y)
            I_x_t[kk] = np.append(I_x_t[kk], ixt)
            I_t_y[key.split('-')[0] + '-o'] = np.append(I_t_y[key.split('-')[0] + '-o'], ity)                
        
    weights_name = ['input-conv1', 'conv1-conv2', 'conv2-dense1', 'dense1-output']
    colors = ['blue', 'green', 'yellow', 'orange', 'red']
    labels = ['1/5', '2/5', '3/5', '4/5', '5/5']
    o = 0
    for i in I_x_t.keys():
        X, T, Y = weights_name[o].split('-')[0], weights_name[o].split('-')[1], 'output'
        intervals = [j for j in range(0, len(I_x_t[i])+int(len(I_x_t[i])/5), int(len(I_x_t[i])/5))]
        plt.title("Information Plane Node Strengths " + weights_name[o])
        plt.xlabel("I({};{})".format(X, T))
        plt.ylabel("I({};{})".format(T, Y))
        for j in range(len(intervals)-1):
            plt.scatter(I_x_t[i][intervals[j]:intervals[j+1]], I_t_y[i.split('-')[0] + '-o'][intervals[j]:intervals[j+1]], color=colors[j], label=labels[j])
        plt.legend(loc='best')
        plt.savefig(dst + 'information_plane_nstrengths_'+weights_name[o]+'.svg')
        plt.savefig(dst + 'information_plane_nstrengths_'+weights_name[o]+'.png')
        plt.pause(0.05)
        o += 1
    
    if show == True:
        plt.show()
    else:
        pass
