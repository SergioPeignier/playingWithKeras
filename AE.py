from keras.layers import Input, Dense, Flatten, Reshape, Activation
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf


class MLA:
    def __init__(self, 
                 input_shape, 
                 layer_sizes, 
                 activations, 
                 regularizers, 
                 stored_weights = None, 
                 optimizer="adam", 
                 loss="mse",
                 metrics=['mae', 'mape']):
        self.model = Sequential()
        self.model.add(Dense(layer_sizes[0],input_shape=(input_shape,),activation=activations[0],activity_regularizer=regularizers[0]))
        for i in range(1,len(layer_sizes)):
            self.model.add(Dense(layer_sizes[i],activity_regularizer=regularizers[i], activation=activations[i]))
        self.model.add(Dense(input_shape))
        if stored_weights is not None: 
            self.model.load_weights(stored_weights)
        self.model.compile(optimizer=optimizer, loss=loss, metrics = metrics)
        
    def fit(self, 
            X_training,  
            batch_size=5, 
            nb_epoch=100, 
            validation_split = 0.01,
            checkpoint_folder="check_points",
            save_each_better_option = False
           ):
        if save_each_better_option:
            filepath_loss =checkpoint_folder+"/weights-improvement-epoch-{epoch:02d}-loss-{loss:.2f}.hdf5"
        else:
            filepath_loss =checkpoint_folder+"/better_loss.hdf5"
        checkpoint_loss = ModelCheckpoint(filepath_loss, monitor='loss', verbose=0, save_best_only=True, mode='auto')
        if save_each_better_option:
            filepath_val_loss =checkpoint_folder+"/weights-improvement-epoch-{epoch:02d}-val_loss-{val_loss:.2f}.hdf5"
        else:
            filepath_val_loss =checkpoint_folder+"/better_val_loss.hdf5"   
        checkpoint_val_loss = ModelCheckpoint(filepath_val_loss, monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint_loss, checkpoint_val_loss]
        self.history = self.model.fit(X_training, X_training, batch_size, nb_epoch, callbacks = callbacks_list, validation_split=validation_split,verbose = 0)
    
    def predict(self,x):
        return self.model.predict(x)
    
    def evaluate(self,x):
        if len(x.shape) == 1:
            x = x.reshape(1,x.shape[0])
        return self.model.evaluate(x,x, verbose = 0)