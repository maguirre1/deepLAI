#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Conv1D
from tensorflow.keras.layers import concatenate, MaxPooling1D, UpSampling1D, Dropout
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K


# define model class
def segnet(input_shape, n_classes, width=16, n_filters=16, dropout_rate=0.01, 
           input_dropout_rate=0.01, pool_size=4, l2_lambda=1e-30, n_blocks=5, 
           batch_normalization=False, crf=None, smooth=False, tanh=False):
    # starting input and dropout
    X_input=Input(shape=input_shape)
    X=Dropout(input_dropout_rate)(X_input)
    
    # down layers
    pools=[]
    for i in range(n_blocks):
        # double convolutional block for each step  
        for j in range(2 + int(i>1)):
            # Defaults are activation=None and stride=1
            X=Conv1D(filters=n_filters*(2**i), kernel_size=width, 
                     padding='same', kernel_initializer='he_normal', 
                     activity_regularizer=regularizers.l2(l2_lambda),
                     name='conv'+str(i+1)+'_down'+str(j+1))(X)
            # optional batchnorm, then ReLU activation and Dropout
            if batch_normalization:
                X=BatchNormalization(center=False)(X)
            X=Activation('tanh' if tanh else 'relu')(X)
            X=Dropout(dropout_rate)(X)
        # scrunch down, save layer
        X=MaxPooling1D(pool_size=pool_size, name='pool'+str(i))(X)
        pools.append(X)
    
    # up layers
    for i in reversed(range(n_blocks)):
        # link up with previous filters, expand back up
        X=concatenate([pools[i], X], axis = -1)
        X=UpSampling1D(size=pool_size, name='up'+str(i))(X)
        for j in range(2 + int(i>1)): 
            X=Conv1D(filters=n_filters*(2**i), kernel_size=width,
                     padding='same', kernel_initializer='he_normal', 
                     activity_regularizer=regularizers.l2(l2_lambda),
                     name='conv'+str(i)+'_up'+str(j))(X)
            if batch_normalization:
                X=BatchNormalization(center=False)(X)
            X=Activation('tanh' if tanh else 'relu')(X)
            X=Dropout(dropout_rate)(X)
            
    # output layer
    if crf is not None: 
        # this is a passed tf2CRF object 
        # - must have loss=crf.loss, metrics=[crf.accuracy] (or similar)
        #   at compile-time (train.py or sandbox.ipynb)
        Y=CRF(n_classes, learn_mode='marginal', test_mode='marginal', activation='softmax')(X)
    elif smooth:
        Y=Conv1D(filters=n_classes, kernel_size=256, padding='same', activation='softmax', name='output_layer')(X)
    else:
        Y=Dense(n_classes, activation='softmax', name='output_layer')(X)

    # done!
    return Model(inputs=X_input, outputs=Y, name='segnet')
