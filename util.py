""" Various Utility Functions """

import tensorflow as tf
import numpy as np
import itertools

#Convert weights from file
def convert_weights(var_list, filename):
    """ Loads official pre-trained YOLOv3 weights (trained on COCO dataset) and
        converts it to TensorFlow format.
        Great explanation of how the weights are structured
        https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/
    Params:
        var_list: list of variables
        filename: name of weight file
    Returns:
        list of tf.assign operations
    """
    with open(filename, 'rb') as file:
        #skip first 3 int32 and 1 int64 of irrelevant headers
        headers = np.fromfile(file, dtype=np.int32, count=5)
        #62,001,757 float32 weights
        weights = np.fromfile(file, dtype=np.float32)

        """ Notes:
            If a batch norm layer is in a convolution block, convolution has no bias
                [bn_bias, bn_weight, bn_running mean, bn_running variance, conv weights]
            If there is no batch norm layer then the convolution layer has bias values to be read
                [conv biases, con weights]
            weights are row major format. TF uses column major
        """

        assign_list = []
        #create generators
        vars = (n for n in var_list)
        weights = (n for n in weights)

        for i in range(75):
            #7th, 15th, 23rd YOLO layers have no batch norm, have bias weights
            if i == 58 or i == 66 or i == 74:
                #add biases [conv biases, conv weights]
                convar = next(vars)
                biasvar = next(vars)
                shape = biasvar.shape.as_list()
                weight = np.asarray(list(itertools.islice(weights, np.prod(shape).item()))).reshape(shape)
                assign_list.append(tf.assign(biasvar, weight))

                shape = convar.shape.as_list()
                weight = np.asarray(list(itertools.islice(weights, np.prod(shape).item()))).reshape((shape[3], shape[2], shape[0], shape[1]))
                weight = np.transpose(weight, (2,3,1,0))
                assign_list.append(tf.assign(convar, weight))
            else:
                convar = next(vars)
                #[gamma, beta, mean, variance]
                bn_vars = list(itertools.islice(vars, 4))
                #[beta, gamma, mean, variance]
                #bn_vars = np.transpose(bn_vars, (1,0,2,3))
                bn_vars = [bn_vars[1],bn_vars[0],bn_vars[2],bn_vars[3]]

                for bvar in bn_vars:
                    shape = bvar.shape.as_list()
                    weight = np.asarray(list(itertools.islice(weights, np.prod(shape).item()))).reshape(shape)
                    assign_list.append(tf.assign(bvar, weight))

                shape = convar.shape.as_list()
                weight = np.asarray(list(itertools.islice(weights, np.prod(shape).item()))).reshape((shape[3], shape[2], shape[0], shape[1]))
                #convert to column major
                weight = np.transpose(weight, (2,3,1,0))
                assign_list.append(tf.assign(convar, weight))

    return assign_list


def box_corners(inputs):
    """ Converts Yolo box detections from center_x, center_y, box_height, box_width to
        top left and bottom right coordinates. Makes it easier to compute IOU
    Params:
        inputs: output tensor from YOLO detection
    Returns:
        Converted tensor [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, classes]
    """
    bx, by, bw, bh, conf, classes = tf.split(inputs, [1,1,1,1,1,-1], axis=-1)

    topx = bx - (bw / 2)
    topy = by - (bh / 2)
    bottomx = bx + (bw / 2)
    bottomy = by + (bh / 2)

    return tf.concat([topx, topy, bottomx, bottomy, conf, classes], axis=-1)


def non_max_supression(inputs, max_output, conf_threshold, iou_threshold):
    """ Class-wise non max suppression. Several articles have reported performance issues
        using tf.image.non_max_supression(). However, i don't feel like implementing my own in
        numpy at the moment.
    Params:
        inputs: tensor of box coordinates and confidence value
        max_output: Scalar integer Tensor for max number of boxes for tf nms function
        conf_threshold: Confidence threshold
        iou_threshold: Intersection over Union threshold
    Returns:
        list of dictionary: key: class, value: list of boxes for class [(box, confidence)]
    """
    #Step: zero out box predictions with confidence less than threshold
    bool_mask = np.expand_dims((inputs[:,:,4] > conf_threshold), axis=-1)
    clean_predictions = inputs * mask

    batch_size = clean_predictions.shape[0]

    #iterate through the batch
    for ind in range(batch_size):
        img_pred = clean_predictions[ind]
        #get only indexes where it's non-zero
        img_pred = img_pred[np.nonzero(img_pred)].reshape(-1, img_pred.shape[-1])
        #get the index of the class with largest classification
        classes = np.argmax(img_pred[:,5:], axis=-1)

        unique_classes = list(set(classes.reshape(-1)))


    pass


#TODO
def draw_boxes():
    pass
