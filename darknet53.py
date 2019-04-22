""" Helpers for constructing Darknet 53 Feature Extractor implemented in TensorFlow """

import tensorflow as tf
import numpy as np

#global
_BATCH_NORM_EPSILON = 1e-05
_BATCH_NORM_DECAY = 0.99
LRELUALPHA = 0.1

#Batch norm helper from Official TF Resnet implementation
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)

#Official fixed padding from TF Resnet implementation
def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(tensor=inputs,
                           paddings=[[0, 0], [pad_beg, pad_end],
                                     [pad_beg, pad_end], [0, 0]])
  return padded_inputs

#Helper for constructing residual block
def residual_block(inputs, numfilters, size, stride=1, mult=1, is_training=False):
    inp = inputs

    #Loop for constructing residual blocks
    for i in mult:
        inp = tf.layers.conv2d(inputs=inp, filters=numfilters * 2, kernel_size=[3, 3], strides=(stride, stride), padding="same")
        #batch normalization
        inp = tf.layers.batch_normalization(
            inputs=inputs, axis=1 if data_format == 'NCHW' else 3,
            momentum=BNDECAY, epsilon=BNEPSILON, scale=True,
            training=is_training, fused=None)
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
            inputs=inputs, axis=1 if data_format == 'NCHW' else 3,
            momentum=BNDECAY, epsilon=BNEPSILON, scale=True,
            training=is_training, fused=None)
    #leaky ReLU
    inp = tf.nn.leaky_relu(inputs, alpha=LRELUALPHA)

    return inp

#Constructs the Darknet 53 model
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
    #route for concatenating upsampled activations before 4th downsample
    route1 = inp
    inp = conv_block(inp, 512, 3, 2, is_training)

    #8x
    inp = residual_block(inp, 256, 1, 1, 8, is_training)
    #route for concatenating upsampled activations before 5th downsample
    route2 = inp
    inp = conv_block(inp, 1024, 3, 2, is_training)

    #4x
    inp = residual_block(inp, 512, 1, 1, 4, is_training)

    #average pooling

    #returns darknet 53 model with routes before 4th and 5th downsample
    return route1, route2, inp
