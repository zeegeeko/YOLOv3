""" Various Utility Functions """

#Load weights from file
def load_weights(vars, filename):
    """ Loads official pre-trained YOLOv3 weights (trained on COCO dataset) and
        converts it to TensorFlow format.
    Params:
        vars: list of variables
        filename: name of weight file
    Returns:
        list of tf.assign operations
    """
    with open(filename, 'rb') as file:



def non_max_supression():
    pass
