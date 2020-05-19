# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:01:11 2020

@author: Emanuele
"""

from __future__ import print_function
import numpy as np
import os
import keras
from argparse import ArgumentParser
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten, Embedding, Reshape
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras import backend as K

# custom seed's range (multiple experiments)
parser = ArgumentParser()
parser.add_argument("-a", "--architecture", dest="architecture", default='fc', type=str,
                    help="Architecture (fc or cnn so far).")
parser.add_argument("-c", "--cut-training", dest="cut_training", action='store_true',
                    help="Enable the selection of the size of the dataset randomly from 1k to 60k")
parser.add_argument("-s", "--seed", dest="seed_range", default=0, type=int,
                    help="Seed range (from n to n+5000).")  
parser.add_argument("-min", "--min", dest="min", default=0.0, type=float,
                    help="Min accuracy values for final models (discard anything below).")
parser.add_argument("-max", "--max", dest="max", default=1.0, type=float,
                    help="Min accuracy values for final models (discard anything above).")
args = parser.parse_args()
architecture = args.architecture
cut_training = args.cut_training
seed_range = args.seed_range
min_range_fin, max_range_fin = args.min, args.max

# import data, once
batch_size = np.random.randint(32,128)
k_size = 12
input_size = k_size*k_size  # power expression for CNNs ;)
num_classes = 2
epochs = 5

# Set unique seed value
# Apparently you may use different seed values at each stage
for seed_value in range(seed_range, seed_range+5000):
    # 1. Set `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)
    # 2. Set `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)
    # 3. Set `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)
    # 4. Set `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.set_random_seed(seed_value)
    # 5. Configure a new global `tensorflow` session
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    # the data, split between train and test sets
    num_words = np.random.randint(20, 1000)
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                          num_words=num_words,
                                                          skip_top=0,
                                                          maxlen=None,
                                                          seed=113,
                                                          start_char=1,
                                                          oov_char=2,
                                                          index_from=3)
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=input_size, dtype='int32', 
                                                         padding='post', truncating='post', value=0)
    x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=input_size, dtype='int32', 
                                                        padding='post', truncating='post', value=0)
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    
    # parameters initializers
    initializers = {}
    initializers['constant-0'] = keras.initializers.Constant(value=0)
    initializers['random-normal-0-0.05'] = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed_value)
    initializers['random-uniform-0.05'] = keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed_value)
    initializers['truncated-normal-0-0.05'] = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=seed_value)
    initializers['variance-scaling-1-faniin-normal'] = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=seed_value)
    initializers['lecun'] = keras.initializers.lecun_uniform(seed=seed_value)
    initializers['orthogonal-1'] =  keras.initializers.Orthogonal(gain=1.0, seed=seed_value)
    initializers['glorot-normal'] = keras.initializers.glorot_normal(seed=seed_value)
    initializers['glorot-uniform'] = keras.initializers.glorot_uniform(seed=seed_value)
    initializers['he-normal'] = keras.initializers.he_normal(seed=seed_value)
    initializers['lecun-normal'] = keras.initializers.he_normal(seed=seed_value)
 
    for key in initializers.keys():
        model = Sequential()
        d = np.random.randint(2,50)
        model.add(Embedding(num_words, d, input_length=input_size))
        if architecture == 'cnn':
            model.add(Reshape((k_size,k_size,d)))
            model.add(Conv2D(np.random.randint(2, 64), kernel_size=(3, 3),activation='relu',kernel_initializer=initializers[key],bias_initializer='zeros'))
            model.add(Flatten())
        elif architecture == 'fc':
            model.add(Flatten())
            model.add(Dense(np.random.randint(2, 32), activation='relu',kernel_initializer=initializers[key], bias_initializer='zeros'))
        model.add(Dense(32, activation='relu',kernel_initializer=initializers[key], bias_initializer='zeros'))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
                      metrics=['accuracy'])
        
        # Save the weights at the first and last iteration
        dst = './results/{}/'.format(architecture)
        save_to_file = True
        """
        print("[CUSTOM-LOGGER]: Saving initial params to file at relative path {}.".format(dst))
        np.save(dst + 'i_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,model.evaluate(x_test, y_test, verbose=0)[1]), 
                np.asarray(model.get_weights()))
        """
        if cut_training is True:
            dataset_size = int(len(x_train))
        else:
            dataset_size = len(x_train)
        # train        
        print("[logger]: Training on {}/{} datapoints.".format(dataset_size, len(x_train)))
        model.fit(x_train[:dataset_size], y_train[:dataset_size],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))
        # test and save
        print("[CUSTOM-LOGGER]: Saving final params to file at relative path {}.".format(dst))                  
        accuracy = model.evaluate(x_test, y_test, verbose=0)[1]
        if accuracy >= 0.975 and accuracy <= 1.0:
            acc_prefix = '0.975-1.0/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.95 and accuracy <= 0.975:
            acc_prefix = '0.95-0.975/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.925 and accuracy <= 0.95:
            acc_prefix = '0.925-0.95/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.9 and accuracy <= 0.925:
            acc_prefix = '0.9-0.925/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.875 and accuracy <= 0.9:
            acc_prefix = '0.875-0.9/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.85 and accuracy <= 0.875:
            acc_prefix = '0.85-0.875/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.825 and accuracy <= 0.85:
            acc_prefix = '0.825-0.85/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.8 and accuracy <= 0.825:
            acc_prefix = '0.8-0.825/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.775 and accuracy <= 0.8:
            acc_prefix = '0.775-0.8/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.75 and accuracy <= 0.775:
            acc_prefix = '0.75-0.775/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.725 and accuracy <= 0.75:
            acc_prefix = '0.725-0.75/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.7 and accuracy <= 0.725:
            acc_prefix = '0.7-0.725/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.675 and accuracy <= 0.7:
            acc_prefix = '0.675-0.7/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.65 and accuracy <= 0.675:
            acc_prefix = '0.65-0.675/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.625 and accuracy <= 0.65:
            acc_prefix = '0.625-0.65/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.6 and accuracy <= 0.625:
            acc_prefix = '0.6-0.625/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.575 and accuracy <= 0.6:
            acc_prefix = '0.575-0.6/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.55 and accuracy <= 0.575:
            acc_prefix = '0.55-0.575/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.525 and accuracy <= 0.55:
            acc_prefix = '0.525-0.55/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy >= 0.5 and accuracy <= 0.525:
            acc_prefix = '0.5-0.525/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
        elif accuracy <= 0.5:
            acc_prefix = '0.475-0.5/'
            if len(next(os.walk(dst+acc_prefix))[2]) <= 2000:
                np.save(dst + acc_prefix + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
