#!/usr/bin/env python
import numpy as np
import keras
from keras import regularizers
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv1D, concatenate
from keras.layers import AveragePooling1D, MaxPooling1D, UpSampling1D, Dropout
from keras.activations import softmax
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
import tensorflow as tf
#import horovod.keras as hvd 


# initialize horovod instance -- this currently only works on galangal
#hvd.init()

# assign GPUs to horovod 
#gpus = tf.config.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
#if gpus:
#    # tf.config in tf <= 1.6
#    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#print(gpus)



# define modeling class
def segnet(input_shape, n_classes):
    filter_width = 16
    n_filters_in_most_shallow_layer = 16
    dropout_rate = 0.01
    dropout_rate_after_input_layer = 0.01
    q=4
    l2_regularization_parameter = 1e-30 # 1e-7 was the greatest value that was still able to learn
    
    X_input = Input(shape=input_shape)

    # this should probably be automated in a smarter way
    dropout1 = Dropout(dropout_rate_after_input_layer)(X_input)
    conv1_down = Conv1D(filters=n_filters_in_most_shallow_layer, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv1_down1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(dropout1)
#    conv1_down = BatchNormalization()(conv1_down)
    conv1_down = Activation('relu')(conv1_down)
    conv1_down = Dropout(dropout_rate)(conv1_down)
    conv1_down = Conv1D(filters=n_filters_in_most_shallow_layer, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv1_down2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv1_down)
#    conv1_down = BatchNormalization()(conv1_down)
    conv1_down = Activation('relu')(conv1_down)
    conv1_down = Dropout(dropout_rate)(conv1_down)
    pool1 = MaxPooling1D(pool_size=q, name='pool1')(conv1_down)
    
    conv2_down = Conv1D(filters=n_filters_in_most_shallow_layer*q, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv2_down1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(pool1)
#    conv2_down = BatchNormalization()(conv2_down)
    conv2_down = Activation('relu')(conv2_down)
    conv2_down = Dropout(dropout_rate)(conv2_down)
    conv2_down = Conv1D(filters=n_filters_in_most_shallow_layer*q, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv2_down2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv2_down)
#    conv2_down = BatchNormalization()(conv2_down)
    conv2_down = Activation('relu')(conv2_down)
    conv2_down = Dropout(dropout_rate)(conv2_down)
    pool2 = MaxPooling1D(pool_size=q, name='pool2')(conv2_down)
    
    conv3_down = Conv1D(filters=n_filters_in_most_shallow_layer*4, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv3_down1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(pool2)
#    conv3_down = BatchNormalization()(conv3_down)
    conv3_down = Activation('relu')(conv3_down)
    conv3_down = Dropout(dropout_rate)(conv3_down)
    conv3_down = Conv1D(filters=n_filters_in_most_shallow_layer*4, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv3_down2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv3_down)
#    conv3_down = BatchNormalization()(conv3_down)
    conv3_down = Activation('relu')(conv3_down)
    conv3_down = Dropout(dropout_rate)(conv3_down)
    conv3_down = Conv1D(filters=n_filters_in_most_shallow_layer*4, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv3_down3', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv3_down)
#    conv3_down = BatchNormalization()(conv3_down)
    conv3_down = Activation('relu')(conv3_down)
    conv3_down = Dropout(dropout_rate)(conv3_down)
    pool3 = MaxPooling1D(pool_size=q, name='pool3')(conv3_down)
    
    conv4_down = Conv1D(filters=n_filters_in_most_shallow_layer*8, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv4_down1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(pool3)
#    conv4_down = BatchNormalization()(conv4_down)
    conv4_down = Activation('relu')(conv4_down)
    conv4_down = Dropout(dropout_rate)(conv4_down)
    conv4_down = Conv1D(filters=n_filters_in_most_shallow_layer*8, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv4_down2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv4_down)
#    conv4_down = BatchNormalization()(conv4_down)
    conv4_down = Activation('relu')(conv4_down)
    conv4_down = Dropout(dropout_rate)(conv4_down)
    conv4_down = Conv1D(filters=n_filters_in_most_shallow_layer*8, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv4_down3', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv4_down)
#    conv4_down = BatchNormalization()(conv4_down)
    conv4_down = Activation('relu')(conv4_down)
    conv4_down = Dropout(dropout_rate)(conv4_down)
    pool4 = MaxPooling1D(pool_size=q, name='pool4')(conv4_down)
        
    conv5_down = Conv1D(filters=n_filters_in_most_shallow_layer*16, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv5_down1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(pool4)
#    conv5_down = BatchNormalization()(conv5_down)
    conv5_down = Activation('relu')(conv5_down)
    conv5_down = Dropout(dropout_rate)(conv5_down)
    conv5_down = Conv1D(filters=n_filters_in_most_shallow_layer*16, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv5_down2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv5_down)
#    conv5_down = BatchNormalization()(conv5_down)
    conv5_down = Activation('relu')(conv5_down)
    conv5_down = Dropout(dropout_rate)(conv5_down)
    conv5_down = Conv1D(filters=n_filters_in_most_shallow_layer*16, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv5_down3', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv5_down)
#    conv5_down = BatchNormalization()(conv5_down)
    conv5_down = Activation('relu')(conv5_down)
    conv5_down = Dropout(dropout_rate)(conv5_down)
    pool5 = MaxPooling1D(pool_size=q, name='pool5')(conv5_down)
    
    up5 = UpSampling1D(size = q, name='up5')(pool5)
    conv5_up = Conv1D(filters=n_filters_in_most_shallow_layer*16, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv5_up1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(up5)
#    conv5_up = BatchNormalization()(conv5_up)
    conv5_up = Activation('relu')(conv5_up)
    conv5_up = Dropout(dropout_rate)(conv5_up)
    conv5_up = Conv1D(filters=n_filters_in_most_shallow_layer*16, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv5_up2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv5_up)
#    conv5_up = BatchNormalization()(conv5_up)
    conv5_up = Activation('relu')(conv5_up)
    conv5_up = Dropout(dropout_rate)(conv5_up)
    conv5_up = Conv1D(filters=n_filters_in_most_shallow_layer*16, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv5_up3', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv5_up)
#    conv5_up = BatchNormalization()(conv5_up)
    conv5_up = Activation('relu')(conv5_up)
    conv5_up = Dropout(dropout_rate)(conv5_up)
    
    conv5_up = concatenate([pool4,conv5_up], axis = -1)
    up4 = UpSampling1D(size = q, name='up4')(conv5_up)
    conv4_up = Conv1D(filters=n_filters_in_most_shallow_layer*8, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv4_up1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(up4)
#    conv4_up = BatchNormalization()(conv4_up)
    conv4_up = Activation('relu')(conv4_up)
    conv4_up = Dropout(dropout_rate)(conv4_up)
    conv4_up = Conv1D(filters=n_filters_in_most_shallow_layer*8, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv4_up2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv4_up)
#    conv4_up = BatchNormalization()(conv4_up)
    conv4_up = Activation('relu')(conv4_up)
    conv4_up = Dropout(dropout_rate)(conv4_up)
    conv4_up = Conv1D(filters=n_filters_in_most_shallow_layer*8, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv4_up3', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv4_up)
#    conv4_up = BatchNormalization()(conv4_up)
    conv4_up = Activation('relu')(conv4_up)
    conv4_up = Dropout(dropout_rate)(conv4_up)
    
    conv4_up = concatenate([pool3,conv4_up], axis = -1)
    up3 = UpSampling1D(size = q, name='up3')(conv4_up)
    conv3_up = Conv1D(filters=n_filters_in_most_shallow_layer*4, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv3_up1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(up3)
#    conv3_up = BatchNormalization()(conv3_up)
    conv3_up = Activation('relu')(conv3_up)
    conv3_up = Dropout(dropout_rate)(conv3_up)
    conv3_up = Conv1D(filters=n_filters_in_most_shallow_layer*4, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv3_up2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv3_up)
#    conv3_up = BatchNormalization()(conv3_up)
    conv3_up = Activation('relu')(conv3_up)
    conv3_up = Dropout(dropout_rate)(conv3_up)
    conv3_up = Conv1D(filters=n_filters_in_most_shallow_layer*4, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv3_up3', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv3_up)
#    conv3_up = BatchNormalization()(conv3_up)
    conv3_up = Activation('relu')(conv3_up)
    conv3_up = Dropout(dropout_rate)(conv3_up)
    
    conv3_up = concatenate([pool2,conv3_up], axis = -1)
    up2 = UpSampling1D(size = q, name='up2')(conv3_up)
    conv2_up = Conv1D(filters=n_filters_in_most_shallow_layer*q, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv2_up1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(up2)
#    conv2_up = BatchNormalization()(conv2_up)
    conv2_up = Activation('relu')(conv2_up)
    conv2_up = Dropout(dropout_rate)(conv2_up)
    conv2_up = Conv1D(filters=n_filters_in_most_shallow_layer*q, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv2_up2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv2_up)
#    conv2_up = BatchNormalization()(conv2_up)
    conv2_up = Activation('relu')(conv2_up)
    conv2_up = Dropout(dropout_rate)(conv2_up)
    
    conv2_up = concatenate([pool1,conv2_up], axis = -1)
    up1 = UpSampling1D(size = q, name='up1')(conv2_up)
    conv1_up = Conv1D(filters=n_filters_in_most_shallow_layer, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv1_up1', activity_regularizer=regularizers.l2(l2_regularization_parameter))(up1)
#    conv1_up = BatchNormalization()(conv1_up)
    conv1_up = Activation('relu')(conv1_up)
    conv1_up = Dropout(dropout_rate)(conv1_up)
    conv1_up = Conv1D(filters=n_filters_in_most_shallow_layer, kernel_size=filter_width, strides=1, activation = None, padding = 'same', kernel_initializer = 'he_normal', name='conv1_up2', activity_regularizer=regularizers.l2(l2_regularization_parameter))(conv1_up)
#    conv1_up = BatchNormalization()(conv1_up)
    conv1_up = Activation('relu')(conv1_up)
    conv1_up = Dropout(dropout_rate)(conv1_up)
    
    softmax_output_layer = Dense(n_classes, activation=None, name='output_layer')(conv1_up)
    softmax_output_layer = Activation('softmax')(softmax_output_layer)

    model = Model(inputs = X_input, outputs = softmax_output_layer, name='segnet_model')
    
    return model


