""" Various Utility Functions """

import numpy as np
import itertools

#Load weights from file
def load_weights(var_list, filename):
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

        #weights for darknet53
        for i in range(52):
            convar = next(vars)
            #[gamma, beta, mean, variance]
            bn_vars = list(itertools.islice(vars, 4))
            #[beta, gamma, mean, variance]
            bn_vars = np.transpose(bn_vars, (1,0,2,3))

            for bvar in bn_vars:
                shape = bvar.shape().as_list()
                weight = list(itertools.islice(weights, np.prod(shape))).reshape(shape)
                assign_list.append(tf.assign(bvar, weight))

            shape = convar.shape().as_list()
            weight = list(itertools.islice(weights, np.prod(shape))).reshape((shape[3], shape[2], shape[0], shape[1]))
            #convert to column major
            weight = np.transpose(weight, (2,3,1,0))
            assign_list.append(tf.assign(convar, weight))

        #weights for Yolo, 7th, 15th, 23rd layers, no batch norm, have bias weights
        for i in range(23):
            if i == 6 or i == 14 or i == 22:
                #add biases [conv biases, conv weights]
                biasvar = next(vars)
                shape = biasvar.shape().as_list()
                weight = list(itertools.islice(weights, np.prod(shape))).reshape(shape)
                assign_list.append(tf.assign(biasvar, weight))

                convar = next(vars)
                shape = convar.shape().as_list()
                weight = list(itertools.islice(weights, np.prod(shape))).reshape((shape[3], shape[2], shape[0], shape[1]))
                weight = np.transpose(weight, (2,3,1,0))
                assign_list.append(tf.assign(convar, weight))
            else:
                convar = next(vars)
                bn_vars = list(itertools.islice(vars, 4))
                bn_vars = np.transpose(bn_vars, (1,0,2,3))

                for bvar in bn_vars:
                    shape = bvar.shape().as_list()
                    weight = list(itertools.islice(weights, np.prod(shape))).reshape(shape)
                    assign_list.append(tf.assign(bvar, weight))

                shape = convar.shape().as_list()
                weight = list(itertools.islice(weights, np.prod(shape))).reshape((shape[3], shape[2], shape[0], shape[1]))
                weight = np.transpose(weight, (2,3,1,0))
                assign_list.append(tf.assign(convar, weight))

    return assign_list





def non_max_supression():
    pass


def generate_boxes():
    pass
