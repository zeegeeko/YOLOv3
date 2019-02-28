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

#Helper for constructing residual block
def residual_block(inputs, numfilters, size, stride=1, mult=1, is_training=False):
    inp = inputs

    #Loop for constructing residual blocks
    for i in mult:
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

    return inp

#Helper for convolution block
def conv_block(inputs, numfilters, size, stride=1, is_training=False):
    inp = inputs
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

    #inputs, numfilters, size, stride=1, mult=1, is_training
    inp = conv_block(inp, 32, 3, 1, is_training)
    inp = conv_block(inp, 64, 3, 2, is_training)

    #1x
    inp = residual_block(inp, 32, 1, 1, 1, is_training)
    inp = conv_block(inp, 128, 3, 2, is_training)

    #2x
    inp = residual_block(inp, 64, 1, 1, 2, is_training)
    inp = conv_block(inp, 256, 3, 2, is_training)

    #8x
    inp = residual_block(inp, 128, 1, 1, 8, is_training)
    route1 = inp
    inp = conv_block(inp, 512, 3, 2, is_training)

    #8x
    inp = residual_block(inp, 256, 1, 1, 8, is_training)
    route2 = inp
    inp = conv_block(inp, 1024, 3, 2, is_training)

    #4x
    inp = residual_block(inp, 512, 1, 1, 4, is_training)

    #average pooling

    #returns darknet 53 model
    return route1, route2, inp
