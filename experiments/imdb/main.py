# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:37:35 2019

@author: Emanuele
Train an LSTM model on the IMDB sentiment classification task.
"""

import keras
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Flatten, Dense, LSTM
from keras.datasets import imdb
from keras import preprocessing

class CustomSaverZero(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        global n_epochs
        if epoch==0: # or save after some epoch, each k-th epoch etc.
            self.model.save("models/model_{}.hd5".format(epoch))

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        global n_epochs
        if epoch==n_epochs-1: # or save after some epoch, each k-th epoch etc.
            self.model.save("models/model_{}.hd5".format(epoch))
            

max_features = 10000
maxlen = 20
n_epochs = 10

embedding_layer = Embedding(1000, 64, trainable=False)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)    


model = Sequential()
model.add(Embedding(10000, 8, input_length=maxlen))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
# train for the first epoch
history = model.fit(x_train, y_train,
                    epochs=1,
                    batch_size=32,
                    validation_split=0.7,
                    callbacks=[CustomSaverZero()])
# training till the network is reliable at classifying inputs
history = model.fit(x_train, y_train,
                    epochs=n_epochs,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[CustomSaver()])  
# test on unseen data
history = model.evaluate(x_test, y_test)    

# save weights from the models (untrained vs trained)
from keras.models import load_model

import numpy as np

init_weights, fin_weights = np.array([]), np.array([])

model = load_model('models/model_0.hd5')
model.save_weights('adj_matrices/init_weights.hdf5')
init_weights = np.fromfile('adj_matrices/init_weights.hdf5', dtype=float)
        
model = load_model('models/model_' + n_epochs-1 + '.hd5')
model.save_weights('adj_matrices/init_weights.hdf5')
fin_weights = np.fromfile('adj_matrices/init_weights.hdf5', dtype=float)

np.save('adj_matrices/init_weights.npy', init_weights)
np.save('adj_matrices/fin_weights.npy', fin_weights)
 