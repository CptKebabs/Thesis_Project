# Main program, will utilise the other subsystems together
import sys
import os
import cv2      #computer vision library
import matplotlib
import numpy as np
from PIL import Image
import open3d as o3d #point cloud stuff
import torch    #Neural network stuff
import Depth_Map
import Point_Cloud as Point_Cloud

FOCAL_LENGTH_X = 400
FOCAL_LENGTH_Y = 400
#do denoising here

#top_mask = predict_image(top_image)# run through our segmentor and save output for later
#bottom_mask = predict_image(bottom_image)

#going to need to generate the depthmap and calculate the scale_factor prior

#top_point_cloud = generate_point_clouds(top_depth, top_direction, top_mask, top_scale_factor) # will take in an image, vector of camera facing direction and 
#bottom_point_cloud = generate_point_clouds(bottom_image, bottom_direction, bottom_mask, bottom_scale_factor) # segmentation mask and will return point cloud corresponding to seaweed

#final_point_cloud = top_point_cloud + bottom_point_cloud

#top_reprojection = reproject(final_point_cloud, top_direction)
#bottom_reprojection = reproject(final_point_cloud, top_direction)

#need to be able to compare original predictions with our ones next


#add the Depth-anything repo to the program path So we can import depth_anything_v2
image_path = "./ImageExtractorOutput/IMG_0865.jpeg"

#GenerateDepthMap.generate_depth_map_as_rgb(image_path)

depth = Depth_Map.generate_depth_map(image_path)
###########################################
point_cloud = Point_Cloud.generate_basic_coloured_point_cloud(image_path, depth, FOCAL_LENGTH_X, FOCAL_LENGTH_Y)
Point_Cloud.render_point_cloud(point_cloud)


################################################################


#Process will be as follows:
    #loop for all pairs of images (first get it working for one then repeat for all in input folder)(
        #preprocess with B3MD (top and bottom)

        #Generate point clouds with DepthAnythingV2 and scale them using a reference object in frame (top and bottom)

        #run segmentation algorithm YOLO(top and bottom)

        #merge point clouds with segmentation algorithm (top and bottom)

        #merge top and bottom coordinate frames (rotation and translation)

        #reproject into each perspective (top and bottom)

        #write point cloud and segmentations to output folder
    #)end loop

    #have something here to compare the original segmentations with the ones this program has created

#Assumptions:
    
    #needs a trained segmentation algorithm

    #will need a way to compare to segmentation algorithm alone

    #images need an object of known size to exist within it to be able to determine scale