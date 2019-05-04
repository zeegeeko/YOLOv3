""" Yolo v3 Class """

import tensorflow as tf
import numpy as np
import darknet53
import detection
import util
#import train

class YOLOv3:

    def __init__(self, numclasses, iou, confidence, data_format):

        self.data_format = data_format
        self.numclasses = numclasses
        self.iou = iou
        self.confidence = confidence


    def detect(self, inputs):
        """
        Returns:
            Yolov3 model
        """
        #normalize inputs
        inputs = inputs / 255
        #list of detection maps at different scales
        detect = []

        #Darket53
        route1, route2, inputs = darknet53(inputs, is_training=False, self.data_format)

        #detection scale 1
        inputs = detection_block(inputs, 512, is_training=False, self.data_format)
        route = inputs
        detect[0] = output_block(inputs, 512, len(priors), self.numclasses, is_training=False, self.data_format)

        #detection scale 2
        inputs = concat_block(route, route2, 256, is_training=False, self.data_format)
        inputs = detection_block(inputs, 256, is_training=False, self.data_format)
        route = inputs
        detect[1] = output_block(inputs, 256, len(priors), self.numclasses, is_training=False, self.data_format)

        #detection scale 3
        inputs = concat_block(route9, route1, 128, is_training=False, self.data_format)
        inputs = detection_block(inputs, 128, is_training=False, self.data_format)
        detect[2] = output_block(inputs, 128, len(priors), self.numclasses, is_training=False, self.data_format)

        #transform predictions
        for i in range(3):
            detect[i] = transform_pred(detect[i], priors, img_size, self.numclasses, self.data_format)

        detections = tf.concat(detect, axis=1)
