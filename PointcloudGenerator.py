#use depthAnythingV2 to generate the depth maps, and convert them to a scaled point cloud here
#https://www.youtube.com/watch?v=vGr8Bg2Fda8
import numpy

#call depthAnythingV2 model on image


def generate_point_clouds(image : numpy.ndarray, direction, mask, scale_factor):
    #generate the depth map or we take old depth map and reconvert it back to depth rather than rgb
    #apply mask
    #convert it to a point cloud
    #scale and rotate as necessary
    return # the point cloud
    




#output scaled point cloud