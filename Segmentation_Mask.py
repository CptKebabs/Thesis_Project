#segmentation mask stuff here

import numpy

def infer_image():
    return #the seaweed prediction

def testing_mask(height, width):
    print(width)
    print(height)
    mask = numpy.zeros((width, height), dtype=bool)
    mask[:int(height):][:int(width):] = True
    return mask