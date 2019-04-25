""" Helpers for constructing the detection layers after darknet53 """

#From YOLOv3 paper https://pjreddie.com/media/files/papers/YOLOv3.pdf
ANCHORS = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]

#Helper Bilinear upsampling
def upsample(inputs, data_format):
""" According to paper, previous feature maps are upsampled by 2x and then
    merged by concatenation. This function upsamples feature maps using bilinear interpolation.
Params:
    inputs: input tensor
    data_format: channel first or channel last
Return:
    upsampled feature map 
"""
    pass
