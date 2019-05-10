""" Various Utility Functions """

import tensorflow as tf
import numpy as np
import itertools
from PIL import Image, ImageDraw, ImageFont

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
        top left and bottom right coordinates (diagonals). Required for NMS
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


def non_max_suppression(inputs, max_output_size, conf_threshold, iou_threshold):
    """ Class-wise non max suppression. Several articles have reported performance issues
        using tf.image.non_max_supression(). However, i don't feel like implementing my own in
        numpy at the moment.
    Params:
        inputs: tensor of box coordinates and confidence value
        max_output: Scalar integer Tensor for max number of boxes for tf nms function
        conf_threshold: Confidence threshold
        iou_threshold: Intersection over Union threshold
    Returns:
        list of dictionaries: key: class, value: list of boxes for class [(box, confidence)]
    """
    batch_size = inputs.get_shape().as_list()[0]
    batch_dicts = []
    numclasses = inputs[:,:,5:].get_shape()[2]

    #iterate through the batch (image)
    for ind in range(batch_size):
        boxes = inputs[ind, :, :]
        #filter out boxes that are not within confidence threshold
        boxes = tf.boolean_mask(boxes, boxes[:,4] > conf_threshold)
        #get the index of the class with largest classification
        classes = tf.argmax(boxes[:,5:], axis=-1)
        classes = tf.expand_dims(classes, axis=-1)
        #concatenate boxes and max class for each box
        boxes = tf.concat([boxes[:,:5], tf.cast(classes, dtype=tf.float32)], axis=-1)

        #Get a list of the unique classes
        #unique_classes = tf.unique(tf.reshape(classes, [-1]))

        class_dict = {}
        for cls in range(numclasses):
            #loop through the unique classes and mask
            cls_mask = tf.equal(boxes[:, 5], tf.cast(cls,dtype=tf.float32))
            if tf.rank(cls_mask) != 0:
                cls_boxes = tf.boolean_mask(boxes, cls_mask)
                box_attr, box_conf, _ = tf.split(cls_boxes, [4, 1, -1], axis=-1)
                box_conf = tf.reshape(box_conf, [-1])
                idxs = tf.image.non_max_suppression(box_attr, box_conf, max_output_size, iou_threshold)
                cls_boxes = tf.gather(cls_boxes, idxs)
                class_dict[cls] = cls_boxes[:, :5]
        batch_dicts.append(class_dict)
    return batch_dicts


#Draws boxes on image using PIL
def draw_boxes(filename, class_names, boxes_dict, input_size):
    """ Draws boxes on image with class name and confidence values
    Params:
        filename: file name of image
        class_names: list of class names
        boxes_dict: dictionary of class to box for image
        input_size: Model input size (needed to recompute coordinates to original size of image)
    Returns:
        Saves image with detections_filename.jpg
    """
    image = Image.open(filename)
    draw = ImageDraw.Draw(image)

    #Iterate through each class in dictionary
    for cls in range(len(class_names)):
        color = tuple(np.random.randint(0, 256, 3))
        if len(boxes_dict[cls]) != 0:
            class_boxes = boxes_dict[cls]
            #Iterate through each box in class
            for box in class_boxes:
                #Convert box dimensions to match proportions of image size
                ratio = np.array(image.size) / np.array(input_size)
                coord = box[:4].reshape(2,2) * ratio
                coord = list(coord.reshape(-1))
                conf = box[4] * 100
                draw.rectangle(coord, outline=color)
                draw.text(coord[:2], '{} {:.2f}%'.format(class_names[cls], conf), fill=color)
    img = image.convert('RGB')
    img.save('./output/detection_' + filename.split('/')[-1])


#Load images
def load_image_batch(file_list, input_size):
    """ Loads images from list of files as an array
    Params:
        file_list: list of filenames
        input_size: (W,H) input size of model
    Returns:
        Numpy array
    """
    batch = []
    for file in file_list:
        img = Image.open(file).resize(input_size)
        img = np.array(img, dtype=np.float32)
        img = np.expand_dims(img[:, :, :3], axis=0)
        batch.append(img)
    return np.concatenate(batch)

#Load class names
def class_names(filename):
    """ Loads file of class names to list
    Params:
        filename: string, filename
    Returns:
        list of class names
    """
    names = []
    with open(filename, 'r') as file:
        for name in file:
            names.append(name.rstrip())
        return names
