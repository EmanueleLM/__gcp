# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 21:48:55 2019

@author: Emanuele
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../../metrics/')

import cumulative as cm
import metrics as mt
import mnist as mnist

# k1.{ofilters, ksize, stride}, k2.{ofilters, ksize, stride}, dense_neurons
custom_p = [[16, 2, 2, 16, 2, 2, 256] for _ in range(3)]

if __name__=="__main__":
    for (i, cp) in zip(range(len(custom_p)), custom_p):        
        acc = mnist.MNIST_model(steps_number=1000, 
                                batch_size=100, 
                                save_to_file=True,
                                params=cp,
                                dst='./MNIST_weights/', 
                                get_m_info=False,
                                get_nodes_strengths_m_info=False)
        
        init = np.load('./MNIST_weights/init_params.npy', allow_pickle=True)
        fin = np.load('./MNIST_weights/fin_params.npy', allow_pickle=True)         
        # nodes mean and variance
        for j in [0,2,4,6]:
            plt.title('[EXPERIMENT {}]: Weights layer {}'.format(i, j))
            plt.hist(init[j].flatten(), bins=50, color='red', alpha=0.5, label="first gen.", normed=True)
            plt.hist(fin[j].flatten(), bins=50, color='blue', alpha=0.5, label="last gen.", normed=True)
            plt.legend(loc='best')
            plt.savefig('./MNIST_results/exp-{0:}-w-layer{1:}-params-{2:}-acc-{3:.3f}.png'.format(i+1, j, cp, acc))
            plt.show()
        # nodes strength, s_in and s_out
        I_shape = (28,28)
        for j in [0,2,4,6]:
            if j <= 2:
                stride = ((custom_p[i][1], custom_p[i][1]) if j==0 else (custom_p[i][5], custom_p[i][5]))
                s_in_i, s_out_i = mt.multichannel_nodes_strength(init[j], I_shape, s=stride, pad='VALID')
                s_in_f, s_out_f = mt.multichannel_nodes_strength(fin[j], I_shape, s=stride, pad='VALID')
                print(I_shape, init[j].shape, custom_p[i][5], s_in_i.shape, s_out_i.shape)
                I_shape = (int(s_out_i.shape[-1]**0.5), int(s_out_i.shape[-1]**0.5))
            else:
                s_in_i, s_out_i = np.sum(init[j], axis=1), np.sum(init[j], axis=0)
                s_in_f, s_out_f = np.sum(fin[j], axis=1), np.sum(fin[j], axis=0)
            plt.title('[EXPERIMENT {}]: In-Out Nodes strength layer {}'.format(i, j))
            plt.hist(s_in_i.flatten(), bins=50, color='red', alpha=0.5, label="s_in first gen.", normed=True)
            plt.hist(s_in_f.flatten(), bins=50, color='yellow', alpha=0.5, label="s_in last gen.",normed=True)
            plt.hist(s_out_i.flatten(), bins=50, color='blue', alpha=0.5, label="s_out first gen.",normed=True)
            plt.hist(s_out_f.flatten(), bins=50, color='green', alpha=0.5,label="s_out last gen.", normed=True)
            plt.legend(loc='best')
            plt.savefig('./MNIST_results/exp-{0:}-ns-layer{1:}-params-{2:}-acc-{3:.3f}.png'.format(i+1, j, cp, acc))
            plt.show() 
        # nodes strength s = s_in + s_out
        prec_i, prec_f = 0., 0.
        I_shape = (28,28)
        for j in [0,2,4,6]:
            if j <= 2:
                stride = ((custom_p[i][1], custom_p[i][1]) if j==0 else (custom_p[i][5], custom_p[i][5]))
                s_in_i, s_out_i = mt.multichannel_nodes_strength(init[j], I_shape, s=stride, pad='VALID')
                s_in_f, s_out_f = mt.multichannel_nodes_strength(fin[j], I_shape, s=stride, pad='VALID')
                I_shape = (int(s_out_i.shape[-1]**0.5), int(s_out_i.shape[-1]**0.5))
            else:
                s_in_i, s_out_i = np.sum(init[j], axis=1), np.sum(init[j], axis=0) 
                s_in_f, s_out_f = np.sum(fin[j], axis=1), np.sum(fin[j], axis=0) 
            s_i = s_in_i.flatten() + prec_i
            prec_i = s_out_i.flatten()
            s_f = s_in_f.flatten() + prec_f
            prec_f = s_out_f.flatten()
            print(s_out_i.shape)
            plt.title('[EXPERIMENT {}]: Nodes strength layer {}'.format(i, j))
            plt.hist(s_i.flatten(), bins=50, color='blue', alpha=0.5, label="s first gen.", normed=True)
            plt.hist(s_f.flatten(), bins=50, color='green', alpha=0.5, label="s first gen.", normed=True)
            plt.legend(loc='best')
            plt.savefig('./MNIST_results/exp-{0:}-fullns-layer{1:}-params-{2:}-acc-{3:.3f}.png'.format(i+1, j, cp, acc))
            plt.show() 
        # cumulative link weights
        for j in [0,2,4,6]:
            plt.title('[EXPERIMENT {}]: Cumulative link weights layer {}'.format(i, j))
            tmp = cm.cumulative_distribution(init[j])
            tmp_f = cm.cumulative_distribution(fin[j])
            plt.hist(init[j].flatten(), bins=50, color='red', alpha=0.5, histtype='step', label="first gen.", cumulative=True, normed=True)
            plt.hist(fin[j].flatten(), bins=50, color='blue', alpha=0.5, histtype='step', label="last gen.", cumulative=True, normed=True)
            plt.legend(loc='best')
            plt.savefig('./MNIST_results/exp-{0:}-clw-layer{1:}-params-{2:}-acc-{3:.3f}.png'.format(i+1, j, cp, acc))
            plt.show()
        # cumulative nodes strength
        for j in [0,2,4,6]:
            plt.title('Cumulative nodes strength input {}, layer {}'.format(i, j))
            i_in, i_out = mt.nodes_strength(init[j])
            f_in, f_out = mt.nodes_strength(fin[j])
            i_in = cm.cumulative_distribution(i_in)
            i_out = cm.cumulative_distribution(i_out)
            f_in = cm.cumulative_distribution(f_in)
            f_out = cm.cumulative_distribution(f_out)            
            plt.hist(i_in, bins=50, color='red', alpha=0.5, histtype='step', label="first gen.", cumulative=True, normed=True)
            plt.hist(f_in, bins=50, color='blue', alpha=0.5, histtype='step', label="last gen.", cumulative=True, normed=True)
            plt.legend(loc='best')
            plt.savefig('./MNIST_results/exp-{0:}-cns-in-layer{1:}-params-{2:}-acc-{3:.3f}.png'.format(i+1, j, cp, acc))
            plt.show()
            plt.title('Cumulative nodes strength output {}, layer {}'.format(i, j))
            plt.hist(i_out, bins=50, color='red', alpha=0.5, histtype='step', cumulative=True, normed=True)
            plt.hist(f_out, bins=50, color='blue', alpha=0.5, histtype='step', cumulative=True, normed=True)
            plt.legend(loc='best')
            plt.savefig('./MNIST_results/exp-{0:}-cns-out-layer{1:}-params-{2:}-acc-{3:.3f}.png'.format(i+1, j, cp, acc))
            plt.show()
        