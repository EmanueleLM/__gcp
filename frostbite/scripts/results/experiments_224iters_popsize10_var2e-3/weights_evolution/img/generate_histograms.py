# cd 'C:\Users\Emanuele\Desktop\Github\__gcp\scripts\results\experiments_224iters_popsize10_var2e-3\weights_evolution'

import numpy as np
import matplotlib.pyplot as plt

weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']

init_weights = np.load('IN_0-0.npy', allow_pickle=True)
fin_weights = np.load('BN_0-last.npy', allow_pickle=True)

for i in range(1,35):
    tmp1 = np.load('IN_'+str(i)+'-0.npy', allow_pickle=True)
    tmp2 = np.load('BN_'+str(i)+'-last.npy', allow_pickle=True)
    for j in range(8):
        init_weights[j] += tmp1[j]
        fin_weights[j] += tmp2[j]

# average along the n best nets
init_weights = np.array([tmp/35. for tmp in init_weights])
fin_weights = np.array([tmp/35. for tmp in fin_weights])

# turn all weights' values positives and normalize weights between 0 and 1
min_ = np.min([[np.min(i) for i in init_weights],
               [np.min(f) for f in fin_weights]])
max_ = np.max([[np.max(i) for i in init_weights],
               [np.max(f) for f in fin_weights]])    
for i in range(len(init_weights)):
    if min_ < 0.:
        init_weights[i] += np.abs(np.min(init_weights[i]))
        fin_weights[i] += np.abs(np.min(fin_weights[i]))
    if max_ != 0.:
        init_weights[i] /= max_
        fin_weights[i] /= max_
    else:
        print("[CUSTOM-LOGGER] Division by zero avoided by adding {} to each parameter in the net".format(1e-6))
        init_weights[i] += 1e-6
        fin_weights[i] += 1e-6
        init_weights[i] /= 1e-6  # prevent division by zero
        fin_weights[i] /= 1e-6

for i in range(8):
    plt.title('Layer: '+ weights_name[i])
    plt.hist(init_weights[i].flatten(), bins=50, color='red', alpha=0.5, label='First Generation', normed=True)
    plt.hist(fin_weights[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation', normed=True)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('hist_'+weights_name[i]+'_best35nets.png')
    plt.savefig('hist_'+weights_name[i]+'_best35nets.svg')
    plt.pause(0.05)
    print("Distance (norm) between two vectors is ", np.linalg.norm(init_weights[i].flatten()-fin_weights[i].flatten()))
plt.show()

# Calculate and plot for each weight Q(w) as the probability that another weight has
#  a value higher than itself
init_Q_w, fin_Q_w = np.array([]), np.array([])
for w in init_weights:
    init_Q_w = np.append(init_Q_w, np.zeros(w.flatten().shape))
    fin_Q_w = np.append(fin_Q_w, np.zeros(w.flatten().shape))

for i in range(len(init_weights)):    
    tmp = init_weights[i].flatten()
    len_w = len(tmp) 
    init_Q_w = np.array([len(tmp[tmp>t])/len_w for t in tmp])
    fin_Q_w = np.array([len(tmp[tmp>t])/len_w for t in tmp])
    
np.save('init_Q_w.npy', init_Q_w)
np.save('fin_Q_w.npy', fin_Q_w)
    
    