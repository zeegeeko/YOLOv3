""" Helpers for constructing the detection layers after darknet53 """

#From YOLOv3 paper https://pjreddie.com/media/files/papers/YOLOv3.pdf
ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]

#Helper Bilinear upsampling
def upsample(inputs, size, data_format):
    """ According to paper, previous feature maps are upsampled by 2x and then
        merged by concatenation. This function upsamples feature maps using bilinear interpolation.
    Params:
        inputs: input tensor
        data_format: channel first or channel last
    Return:
        upsampled feature map
    """
    inputs = tf.image.resize_bilinear(inputs, (new_height, new_width), align_corners=False, name=None)

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


def concat_block(inputs, numfilters, is_training, data_format):
    """ Concatenates route from previous detector layers to route from
        Darknet53 layers
    Params:
        inputs: input tensor
        numfilters: number of filters for convolution
        is_training: bool, true if in training mode
        data_format: channel first or channel last
    Returns:
        Concatenated routes
    """

    #Upsample previous feature maps before concatenation

    pass

def output_block(inputs, numfilters, is_training, data_format):
    """ Last layers, DBL block and output convolution layers
    Params:
        inputs: input tensor
        numfilters: number of filters for convolution
        is_training: bool, true if in training mode
        data_format: channel first or channel last
    Returns:
        outputs
    """

    #last DBL block
    inputs = conv_block(inputs, 2 * numfilters, 3, 1, is_training, data_format)

    #output layer

    pass
