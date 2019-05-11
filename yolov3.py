""" Yolo v3 Class """

import tensorflow as tf
import numpy as np
from darknet53 import *
from detection import *
from util import *
#import train


class YOLOv3:

    def __init__(self, img_size, numclasses, priors, iou, confidence, max_output_size, data_format):

        self.img_size = img_size
        self.data_format = data_format
        self.numclasses = numclasses
        self.priors = priors
        self.iou = iou
        self.confidence = confidence
        self.max_output_size = max_output_size

    def detect(self, inputs):
        """
        Returns:
            Yolov3 model
        """
        with tf.variable_scope('yolov3'):
            #normalize inputs
            inputs = inputs / 255
            #list of detection maps at different scales
            detect = []

            #Darket53
            route1, route2, inputs = darknet53(inputs, False, self.data_format)

            #detection scale 1
            inputs = detection_block(inputs, 512, False, self.data_format)
            route = inputs
            detect.append(output_block(inputs, 512, 3, self.numclasses, False, self.data_format))

            #detection scale 2
            inputs = concat_block(route, route2, 256, False, self.data_format)
            inputs = detection_block(inputs, 256, False, self.data_format)
            route = inputs
            detect.append(output_block(inputs, 256, 3, self.numclasses, False, self.data_format))

            #detection scale 3
            inputs = concat_block(route, route1, 128, False, self.data_format)
            inputs = detection_block(inputs, 128, False, self.data_format)
            detect.append(output_block(inputs, 128, 3, self.numclasses, False, self.data_format))

            #transform predictions
            for i, j in enumerate(range(2,-1,-1)):
                detect[i] = transform_pred(detect[i], self.priors[j * 3:(j * 3) + 3], self.img_size, self.numclasses, self.data_format)

            #convert box coordinates to diagonals
            detections = box_corners(tf.concat(detect, axis=1))

            #Non-max suppression
            return non_max_suppression(detections, self.max_output_size, self.confidence, self.iou)
