# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 22:01:11 2020

@author: Emanuele
"""

from __future__ import print_function
import keras
from argparse import ArgumentParser
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras import backend as K

# custom seed's range (multiple experiments)
parser = ArgumentParser()
parser.add_argument("-a", "--architecture", dest="architecture", default='fc', type=str,
                    help="Architecture (fc or cnn so far).")
parser.add_argument("-c", "--cut-training", dest="cut_training", action='store_true',
                    help="Enable the selection of the size of the dataset randomly from 1k to 60k")
parser.add_argument("-s", "--seed", dest="seed_range", default=0, type=int,
                    help="Seed range (from n to n+10).")  
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
batch_size = 128
num_classes = 10
epochs = 1
# input image dimensions
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Set unique seed value
# Apparently you may use different seed values at each stage
for seed_value in range(seed_range, seed_range+10):
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
        if architecture == 'cnn':
            model.add(Conv2D(np.random.randint(2, 8), kernel_size=(3, 3),activation='relu',input_shape=input_shape,kernel_initializer=initializers[key],bias_initializer='zeros'))
            model.add(Conv2D(np.random.randint(2, 8), (3, 3), activation='relu', kernel_initializer=initializers[key], bias_initializer='zeros'))
            model.add(Flatten())
        elif architecture == 'fc':
            model.add(Flatten())
            model.add(Dense(np.random.randint(2, 64), activation='relu',kernel_initializer=initializers[key], bias_initializer='zeros'))
            model.add(Dense(np.random.randint(2, 64), activation='relu',kernel_initializer=initializers[key], bias_initializer='zeros'))
        model.add(Dense(128, activation='relu',kernel_initializer=initializers[key], bias_initializer='zeros'))
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
            dataset_size = int(np.random.rand()*len(x_train))
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
        if accuracy >= min_range_fin and accuracy <= max_range_fin:
            np.save(dst + 'f_seed-{}_init-{}_score-{:.3f}.npy'.format(seed_value,key,accuracy),np.asarray(model.get_weights()))
