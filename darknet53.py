""" Helpers for constructing Darknet 53 Feature Extractor implemented in TensorFlow """

import tensorflow as tf
import numpy as np

#global
BNEPSILON = 1e-05
BNDECAY = 0.99
LRELUALPHA = 0.1

#Fixed padding
def fixed_padding(inputs):
    #Need to implement fixed padding if kernel_size > 1
    pass

#Constructs Blocks of Convolution Layers with residuals.
def construct_layers(inputs, numfilters, size, stride=1, mult=1, block=True, is_training=False):
    inp = inputs

    #Loop for constructing residual blocks
    for i in mult:

        if block:
            inp = tf.layers.conv2d(inputs=inp, filters=numfilters * 2, kernel_size=[3, 3], strides=(stride, stride), padding="same")
            #batch normalization
            inp = tf.layers.batch_normalization(
                    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
                    momentum=BNDECAY, epsilon=BNEPSILON,
                    scale=True, training=is_training)
            #leaky ReLU
            inp = tf.nn.leaky_relu(inputs, alpha=LRELUALPHA)
            #Residual
            inp = inp + inputs
        else:
            inp = tf.layers.conv2d(inputs=inp, filters=numfilters, kernel_size=[size, size], strides=(stride, stride), padding="same")
            #batch normalization
            inp = tf.layers.batch_normalization(
                    inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
                    momentum=BNDECAY, epsilon=BNEPSILON,
                    scale=True, training=is_training)
            #leaky ReLU
            inp = tf.nn.leaky_relu(inputs, alpha=LRELUALPHA)
    return inp

#Constructs the Darknet 53 network
def darknet53(inputs, is_training):
    inp = inputs

    #inputs, numfilters, size, stride=1, mult=1, block=True
    inp = construct_layers(inp, 32, 3, 1, 1, False, is_training)
    inp = construct_layers(inp, 64, 3, 2, 1, False, is_training)

    #1x
    inp = construct_layers(inp, 32, 1, 1, 1, True, is_training)
    inp = construct_layers(inp, 128, 3, 2, 1, False, is_training)

    #2x
    inp = construct_layers(inp, 64, 1, 1, 2, True, is_training)
    inp = construct_layers(inp, 256, 3, 2, 1, False, is_training)

    #8x
    inp = construct_layers(inp, 128, 1, 1, 8, True, is_training)
    inp = construct_layers(inp, 512, 3, 2, 1, False, is_training)

    #8x
    inp = construct_layers(inp, 256, 1, 1, 8, True, is_training)
    inp = construct_layers(inp, 1024, 3, 2, 1, False, is_training)

    #4x
    inp = construct_layers(inp, 512, 1, 1, 4, True, is_training)

    #average pooling

    #returns darknet 53 model
    return inp
