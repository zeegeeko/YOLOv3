""" Helpers for constructing Darknet 53 Feature Extractor implemented in TensorFlow """

import tensorflow as tf
import numpy as np

#Constructs Blocks of Convolution Layers with residuals.
def construct_layers(inputs, numfilters, size, stride=1, mult=1, block=True):
    inp = inputs

    for i in mult:
        inp = tf.layers.conv2d(inputs=inp, filters=numfilters, kernel_size=[size, size], strides=(stride, stride), padding="same")

        if block:
            inp = tf.layers.conv2d(inputs=inp, filters=numfilters * 2, kernel_size=[3, 3], strides=(stride, stride), padding="same")
            #Residual (Concatenation)
            inp = inp + inputs

    return inp

#Constructs the Darknet 53 network
def darknet53_network(inputs):
    inp = inputs

    #inputs, numfilters, size, stride=1, mult=1, block=True
    inp = construct_layers(inp, 32, 3, 1, 1, False)
    inp = construct_layers(inp, 64, 3, 2, 1, False)

    #1x
    inp = construct_layers(inp, 32, 1, 1, 1, True)
    inp = construct_layers(inp, 128, 3, 2, 1, False)

    #2x
    inp = construct_layers(inp, 64, 1, 1, 2, True)
    inp = construct_layers(inp, 256, 3, 2, 1, False)

    #8x
    inp = construct_layers(inp, 128, 1, 1, 8, True)
    inp = construct_layers(inp, 512, 3, 2, 1, False)

    #8x
    inp = construct_layers(inp, 256, 1, 1, 8, True)
    inp = construct_layers(inp, 1024, 3, 2, 1, False)

    #4x
    inp = construct_layers(inp, 512, 1, 1, 4, True)

    #average pooling

    #returns darknet 53 model
    return inp
