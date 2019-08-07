## Tutto questo è da fare su tutte le reti neurali ottime, mediando sui pesi di ciascun layer
# 1. generare tutti i pesi finali delle reti ottime
# 2. mediare su ciascun layer, per ogni rete neurale
# 3. plot istogrammi pesi iniziali vs. media pesi finali

# cd 'C:\Users\Emanuele\Desktop\Github\__gcp\scripts\results\experiments_224iters_popsize10_var2e-3\elite'

import numpy as np
import matplotlib.pyplot as plt

# [ANALYSIS 1]: check variation for a single set of weights along all the generations (first ge. vs. all the others)
weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']
weights_shapes = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]
n_gen = 177
selected_weigths = 4

weights = np.zeros(shape=[n_gen, *weights_shapes[selected_weigths]])
for i in range(n_gen):
	weights[i] = np.load('adj_gen_'+str(i)+'.npy', allow_pickle=True)[selected_weigths]

for i in range(1, n_gen):
    plt.title('Layer'+ weights_name[selected_weigths] +' , generation number ' + str(i))
    plt.hist(weights[0].flatten(), bins=50, color='red', alpha=0.5, label='Initial', normed=True)
    plt.hist(weights[i].flatten(), bins=50, color='blue', alpha=0.5, label='Gen'+str(i), normed=True)
    plt.legend(loc='upper right')
    plt.pause(0.05)
plt.show()

# # [ANALYSIS 2]: check variation of weights, first gen. vs. the last one, along all the layers
weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']
weights_shapes = [(8, 8, 4, 16), (1, 1, 1, 16), (4, 4, 16, 32), (1, 1, 1, 32), (3872, 256), (256,), (256, 18), (18,)]
n_gen = 177

initial_weights = np.load('adj_gen_0.npy', allow_pickle=True)
final_weights = np.load('adj_gen_'+str(n_gen)+'.npy', allow_pickle=True)

for i in range(len(initial_weights)):
   plt.title('Initial and final distributions for weights {}'.format(weights_name[i]))
   plt.hist(initial_weights[i].flatten(), bins=50, color='red', alpha=0.5, label='Initial', normed=True)
   plt.hist(final_weights[i].flatten(), bins=50, color='blue', alpha=0.5, label='Final', normed=True)
   plt.legend(loc='upper right')
   plt.pause(0.05)
plt.show()

# Extract receptive fields from the kernels (assuming the input is a rgba image)
kernels_shapes = [(8, 8, 4, 16), (4, 4, 16, 32)]
kernels_names = ['conv1', 'conv2']

init_kernel1 = initial_weights[0]
fin_kernel1 = final_weights[0]
init_kernel2 = initial_weights[2]
fin_kernel2 = final_weights[2]

# first kernel extraction
init_rec_field1 = np.zeros(shape=(84, 84, 4, 16))
fin_rec_field1 = np.zeros(shape=(84, 84, 4, 16))

for i in range(16):
	for n in range(0, 80, 4):
		for m in range(0, 80, 4):
			init_rec_field1[n:n+8,m:m+8,:,i] += init_kernel1[:,:,:,i]
			fin_rec_field1[n:n+8,m:m+8,:,i] += fin_kernel1[:,:,:,i]

for i in range(16):
    plt.title('Initial receptive fields, n°' + str(i))
    plt.imshow(init_rec_field1[:,:,:,i])
    plt.pause(0.05)
    plt.title('Final receptive fields, n°' + str(i))
    plt.imshow(fin_rec_field1[:,:,:,i])
    plt.pause(0.05)
plt.show()

# second kernel extraction
init_rec_field2 = np.zeros(shape=(21,21,16,32))
fin_rec_field2 = np.zeros(shape=(21,21,16,32))

for i in range(32):
	for n in range(0, 18, 2):
		for m in range(0, 18, 2):
			init_rec_field2[n:n+4,m:m+4,:,i] += init_kernel2[:,:,:,i]
			fin_rec_field2[n:n+4,m:m+4,:,i] += fin_kernel2[:,:,:,i]
				
for i in range(32):
    plt.title('Initial receptive fields, n°' + str(i))
    plt.imshow(init_rec_field2[:,:,:,i])
    plt.pause(0.05)
    plt.title('Final receptive fields, n°' + str(i))
    plt.imshow(fin_rec_field2[:,:,:,i])
    plt.pause(0.05)
plt.show()
