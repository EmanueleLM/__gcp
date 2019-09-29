# cd 'C:\Users\Emanuele\Desktop\Github\__gcp\scripts\results\experiments_224iters_popsize10_var2e-3\weights_evolution'

def hist_weights_mean_variance(init_weights, fin_weights, dst='', show=True):
    """
        Plot histograms of weights' mean and variance, for each nn layer
    """

    import numpy as np
    import matplotlib.pyplot as plt
    
    weights_name = ['conv1', 'bconv1', 'conv2', 'bconv2', 'dense1', 'bdense1', 'dense2', 'bdense2']
       
    for i in range(8):
        plt.title('Layer: '+ weights_name[i])
        plt.hist(init_weights[i].flatten(), bins=50, color='red', alpha=0.5, label='First Generation', normed=True)
        plt.hist(fin_weights[i].flatten(), bins=50, color='blue', alpha=0.5, label='Last Generation', normed=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig(dst + 'hist_'+weights_name[i]+'_best35nets.png')
        plt.savefig(dst + 'hist_'+weights_name[i]+'_best35nets.svg')
        plt.pause(0.05)
        print("[CUSTOM-LOGGER]: Distance (norm) between two vectors is {}.".format(np.linalg.norm(init_weights[i].flatten()-fin_weights[i].flatten())))
    
    if show == True:
        plt.show()
    else:
        pass
    