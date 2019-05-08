""" Helpers for constructing the detection layers after darknet53 """

import tensorflow as tf
from darknet53 import *

#From YOLOv3 paper https://pjreddie.com/media/files/papers/YOLOv3.pdf, bounding box priors
#dervived using k-means clustering on COCO dataset
ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]

#Helper Nearest Neighbor upsampling
def upsample(inputs, shape, data_format):
    """ According to paper, previous feature maps are upsampled by 2x and then
        merged by concatenation. This helper upsamples feature maps using Nearest
        Neighbor interpolation. Changed from Bilinear interpolation citing
        https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe
    Params:
        inputs: input tensor
        shape: route shape
        data_format: channel first or channel last
    Return:
        upsampled feature map
    """
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    if data_format == 'channels_first':
        new_height = shape[3]
        new_width = shape[2]
    else:
        new_height = shape[2]
        new_width = shape[1]

    inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

    # back to NCHW if needed
    if data_format == 'channels_first':
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    inputs = tf.identity(inputs, name='upsampled')
    return inputs


# x5 DBL (Darknet/Batch Norm/Leaky RELU) blocks
def detection_block(inputs, numfilters, is_training, data_format):
    """ Creates blocks of 5 conv2d blocks with batch norm and leaky relu for
        Yolo detection layers
    Params:
        inputs: input tensor
        is_training: bool, true if in training mode
        data_format: channel first or channel last
    Return:
        block output (which is also route to detectors at different scales)
    """
    #kernel size 1
    inputs = conv_block(inputs, numfilters, 1, 1, is_training, data_format)
    #kernel size 3
    inputs = conv_block(inputs, 2 * numfilters, 3, 1, is_training, data_format)
    #kernel size 1
    inputs = conv_block(inputs, numfilters, 1, 1, is_training, data_format)
    #kernel size 3
    inputs = conv_block(inputs, 2 * numfilters, 3, 1, is_training, data_format)
    #kernel size 1
    inputs = conv_block(inputs, numfilters, 1, 1, is_training, data_format)

    return inputs


def concat_block(inputs, route, numfilters, is_training, data_format):
    """ Concatenates route from previous detector layers to route from
        Darknet53 layers
    Params:
        inputs: input tensor
        route: route from darknet to be concatenated
        numfilters: number of filters for convolution
        is_training: bool, true if in training mode
        data_format: channel first or channel last
    Returns:
        Concatenated routes
    """
    #DBL
    inputs = conv_block(inputs, numfilters, 1, 1, is_training, data_format)
    #Upsample previous feature maps before concatenation. Match route shape
    inputs = upsample(inputs, route.get_shape().as_list(), data_format)
    #concatenation
    return tf.concat([inputs, route], axis=(1 if data_format is "channels_first" else 3))


def output_block(inputs, numfilters, numpriors, numclasses, is_training, data_format):
    """ Last layers, DBL block and output convolution layers.
    Params:
        inputs: input tensor
        numfilters: number of filters for convolution
        numpriors: number of priors (anchors)
        numclasses: number of prediction classes (80 for COCO)
        is_training: bool, true if in training mode
        data_format: channel first or channel last
    Returns:
        output feature map
    """
    #last DBL block
    inputs = conv_block(inputs, 2 * numfilters, 3, 1, is_training, data_format)

    #output convolution layer [N, numpriors*(4+1+numclasses), W, H]
    return tf.layers.conv2d(inputs, filters=numpriors * (5 + numclasses),
                                kernel_size=1, strides=1,
                                data_format=data_format, use_bias=True,
                                bias_initializer=tf.zeros_initializer())


# Transforms prediction from Yolo
def transform_pred(predictions, priors, img_size, numclasses, data_format):
    """ Transforms convolution feature map from Yolo. According to YOLOv3 paper the output
        tensor is 3-d of 3 box predictions per scale of N × N × [3 * (4 + 1 + 80)] for
        the 4 bounding box offsets,1 objectness prediction, and 80 (COCO dataset) class predictions.
    Params:
        predictions: output tensor from Yolo
        priors: list of bounding box priors (Anchors)
        img_size: tuple (W, H)
        numclasses: number of prediction classes (80 for COCO)
        data_format: channel first or channel last
    Returns:
        transformed bounding box predictions and confidence score as 2-d tensor
        b_x, b_y, b_w, b_h, sigma(t_o), classes
    """

    shape = predictions.get_shape().as_list()
    if data_format is 'channels_first':
        predictions = tf.transpose(predictions, [0,2,3,1])

    #get grid dimensions (H, W), channels_first is [N,C,H,W]
    grid_dim = shape[2:] if data_format is 'channels_first' else shape[1:3]
    #number of grid cells
    numcells = grid_dim[0] * grid_dim[1]

    predictions = tf.reshape(predictions, [-1, len(priors) * numcells, 5 + numclasses])
    """
        Notes from YOLOv3 paper
        t_x, t_y, t_w, t_h, t_o = parameter predictions from Yolo
        b_x = \sigma(t_x) + c_x  where c_x, c_y are the top left corner of grid cell of prior
        b_y = \sigma(t_y) + c_y
        b_w = p_{w}e^{t_w}     where p_w is the prior width
        b_h = p_{h}e^{t_h}     where p_h is the prior height
        \sigma(t_o) = box confidence score
        c_x, c_y, p_w, p_h  normalized by image width, height
        image size is larger than detection map by a factor of stride
    """
    txty, thtw, to, classes = tf.split(predictions, [2,2,1,numclasses], axis=-1)

    #compute the offset from grid cell corner of priors (c_x, c_y)
    gridx = tf.range(grid_dim[0], dtype=tf.float32)
    gridy = tf.range(grid_dim[1], dtype=tf.float32)
    cx, cy = tf.meshgrid(gridx, gridy)
    cx = tf.reshape(cx, (-1, 1))
    cy = tf.reshape(cy, (-1, 1))
    cxcy = tf.concat([cx, cy], axis=-1)
    cxcy = tf.tile(cxcy, [1, len(priors)])
    cxcy = tf.reshape(cxcy, [1, -1, 2])

    stride = (img_size[0] // grid_dim[0], img_size[1] // grid_dim[1])
    priors = tf.tile(priors, [numcells, 1])
    #bxby needs to be multiplied by a factor of stride
    bxby = (tf.nn.sigmoid(txty) + cxcy) * stride
    bhbw = priors * tf.exp(thtw)
    confidence = tf.nn.sigmoid(to)
    classes = tf.nn.sigmoid(classes)

    return tf.concat([bxby, bhbw, confidence, classes], axis=-1)
