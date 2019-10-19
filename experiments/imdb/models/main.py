# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:37:35 2019

@author: Emanuele
Train an LSTM model on the IMDB sentiment classification task.
"""

import keras
from keras.layers import Embedding
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, LSTM
from keras.datasets import imdb
from keras import preprocessing

class CustomSaverZero(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        global n_epochs
        if epoch==0: # or save after some epoch, each k-th epoch etc.
            self.model.save("saved_models/model_{}.hd5".format(epoch))

class CustomSaverLast(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        global n_epochs
        if epoch==n_epochs-1: # or save after some epoch, each k-th epoch etc.
            self.model.save("saved_models/model_{}.hd5".format(epoch))
            
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, logs={}):
        global n_epochs
        input_ = model.input
        outputs = [layer.output for layer in model.layers] 
        self.model.save("tmp_weights/model_{}.hd5".format(epoch))
            

max_features = 10000
maxlen = 150
n_epochs = 15
validation_split, validation_limit, validation_step = .95, .2, .05

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)    

model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen, trainable=True))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=SGD(lr=.2), loss='binary_crossentropy', metrics=['acc'])
model.summary()

# train for the first epoch
history = model.fit(x_train, y_train,
                    epochs=1,
                    batch_size=32,
                    validation_split=0.95,
                    callbacks=[CustomSaverZero(), CustomSaver()])
# training till the network is reliable at classifying inputs
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=32,
                    validation_split=0.5,
                    callbacks=[CustomSaverLast(), CustomSaver()])  
# test on unseen data
history = model.evaluate(x_test, y_test)    

# save weights from the models (untrained vs trained)
from keras.models import load_model

import numpy as np

model = load_model('saved_models/model_0.hd5')
np.save('../results/adj_matrices/init_weights.npy', np.asarray(model.get_weights()))
            
model = load_model('saved_models/model_' + str(n_epochs-1) + '.hd5')
np.save('../results/adj_matrices/fin_weights.npy', np.asarray(model.get_weights()))

# store each weight into a .npy file
for e in range(n_epochs):
    model = load_model('tmp_weights/model_' + str(n_epochs-1) + '.hd5')
    np.save('tmp_weights/weights_' + str(e) + '.npy', np.asarray(model.get_weights()))

# compute the statistics to get the mutual information and the information plane
# dictionary used to estimate mutual info and the information plane trajectories
layers_to_estimate = [l1, l2, l3, output]  # should add the input
# I(x,t)
I_x_t = {'i-l1': np.array([]), 'l1-l2': np.array([]), 'l2-o': np.array([])}
# I(t,y)
I_t_y = {'i-o': np.array([]), 'l1-o': np.array([]), 'l2-o': np.array([])}
tf_shapes = {'i': (batch_size, 28*28), 
             'l1': (batch_size, np.prod(l1.get_shape().as_list()[1:])),
             'l2': (batch_size, np.prod(l2.get_shape().as_list()[1:])),
             'o': (batch_size, np.prod(output.get_shape().as_list()[1:]))}

for e in range(n_epochs):
    pass 