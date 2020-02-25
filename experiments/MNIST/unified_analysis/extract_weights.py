# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 20:15:12 2020

@author: Emanuele
"""
import glob
import numpy as np

def filename_gt(path, accuracy_threshold, prefix='f'):
    files_ = []
    for g in glob.glob(path + prefix + '*.npy'):
        if float('0.'+g.split('.npy')[0].split('0.')[-1])>=accuracy_threshold[0] and float('0.'+g.split('.npy')[0].split('0.')[-1])<accuracy_threshold[1]:
            files_.append(g)
    return files_

if __name__ == "__main__":
    topology = 'cnn'
    init, fin = np.zeros(shape=(128,10)), np.zeros(shape=(128,10)) 
    init_acc_le = (0., .15)
    fin_acc_ge = (0.25, .75)
    for i in filename_gt('results/{}/'.format(topology), init_acc_le, prefix='i'):
        tmp = np.load(i, allow_pickle=True)
        init += tmp[-1]+tmp[-2]  # matrix and bias
    for f in filename_gt('results/{}/'.format(topology), fin_acc_ge, prefix='f'):
        tmp = np.load(f, allow_pickle=True)
        fin += tmp[-1]+tmp[-2]  # matrix and bias
    # average the values
    init /= len(filename_gt('results/{}/'.format(topology), init_acc_le, prefix='i'))
    fin /= len(filename_gt('results/{}/'.format(topology), fin_acc_ge, prefix='f'))
    # save weights
    np.save('./results/{}_weights_npy/init_weights_acc-{}-{}.npy'.format(topology, init_acc_le[0], init_acc_le[1]), init)
    np.save('./results/{}_weights_npy/fin_weights_acc-{}-{}.npy'.format(topology, fin_acc_ge[0], fin_acc_ge[1]), fin)
