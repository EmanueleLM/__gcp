# cd 'C:\Users\Emanuele\Desktop\Github\__gcp\scripts\results\experiments_224iters_popsize10_var2e-3\weights_evolution'

import numpy as np
import matplotlib.pyplot as plt

weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']

init_weights = np.load('analysis_adj_fin0.npy', allow_pickle=True)
fin_weights = np.load('analysis_adj_fin0.npy', allow_pickle=True)

for i in range(1,35):
    tmp1 = np.load('analysis_adj_init'+str(i)+'.npy', allow_pickle=True)
    tmp2 = np.load('analysis_adj_fin'+str(i)+'.npy', allow_pickle=True)
    for j in range(8):
        init_weights[j] += tmp1[j]
        fin_weights[j] += tmp2[j]

# average along the n best nets
init_weights = np.array([tmp/35. for tmp in init_weights])
fin_weights = np.array([tmp/35. for tmp in fin_weights])

for i in range(8):
    plt.title('Layer '+ weights_name[i] +' , generation number ' + str(i))
    plt.hist(init_weights[i].flatten(), bins=50, color='red', alpha=0.5, label='First Generation', normed=True)
    plt.hist(fin_weights[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation', normed=True)
    plt.xlabel('Parameters values')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.savefig('hist_'+weights_name[i]+'_best35nets.png')
    plt.savefig('hist_'+weights_name[i]+'_best35nets.svg')
    plt.pause(0.05)
    print("Distance (norm) between two vectors is ", np.linalg.norm(init_weights[i].flatten()-fin_weights[i].flatten()))
plt.show()