import tensorflow as tf

from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Input,LeakyReLU, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, UpSampling2D, Add, Concatenate, SeparableConv2D, GlobalAveragePooling2D, Multiply, Reshape, SpatialDropout2D, DepthwiseConv2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras import regularizers

from tensorflow.keras.optimizers import Adam, RMSprop, SGD

from tensorflow.keras import initializers

from tensorflow.keras.models import load_model,save_model

from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,TerminateOnNaN,ReduceLROnPlateau, TensorBoard

from tensorflow.keras import backend as K

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input

import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math


def conv_block(x, channels, kernel_size=3, stride=1, weight_decay=1e-4, dropout_rate=None,act='r'):
    kr = regularizers.l2(weight_decay)
    ki = 'he_normal'

    x = Conv2D(channels, (kernel_size, kernel_size), kernel_initializer=ki, strides=(stride, stride),
               use_bias=False, padding='same', kernel_regularizer=kr)(x)
    x = BatchNormalization()(x)


    if (act == 'l'):
        x = LeakyReLU(alpha=0.1)(x)

    if (act == 'r'):
        #x = Activation('relu')(x)
        x = tf.keras.activations.relu(x,max_value=6)

    if (act == 's'):
        x = Activation('sigmoid')(x)

    if dropout_rate != None and dropout_rate != 0.:
        x = Dropout(dropout_rate)(x)
        # x = SpatialDropout2D(dropout_rate)(x)
    return x


def separable_conv_block(x, channels, kernel_size=3, stride=1, weight_decay=5e-4, dropout_rate=None, act='r',res=False):
    ki = initializers.RandomNormal(mean=0.0, stddev=0.2)
    kr = regularizers.l2(weight_decay)

    inp=x
    x = SeparableConv2D(channels, (kernel_size, kernel_size), kernel_initializer=ki,
                      strides=(stride, stride), use_bias=False, padding='same',
                      kernel_regularizer=kr)(x)


    x = BatchNormalization()(x)

    if (act == 'l'):
        x = LeakyReLU(alpha=0.1)(x)

    if (act == 'r'):
        #x = Activation('relu')(x)
        x=tf.keras.activations.relu(x,max_value=6)

    if (act == 's'):
        x = Activation('sigmoid')(x)

    if(res==True and stride == 1 and inp.shape[-1]==x.shape[-1]):
        x = Add()([x,inp])

    if dropout_rate != None and dropout_rate != 0.:
        # x = Dropout(dropout_rate)(x)
        x = SpatialDropout2D(dropout_rate)(x)

    return x
