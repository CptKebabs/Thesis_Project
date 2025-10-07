# Main program, will utilise the other subsystems together
import sys
import os
import cv2      #computer vision library
import matplotlib
import numpy as np
from PIL import Image
import open3d as o3d #point cloud stuff
import torch    #Neural network stuff

#This repo imports
import Depth_Map
import Point_Cloud
import Segmentation_Mask


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

#setup

#read the file to determine file paths and scale factor

ref_obj_points, depth_values = Depth_Map.read_scale_file("ImageExtractorOutput/HouseHD_500.scale")#reads as string
print(ref_obj_points)
print(depth_values)
#now test the more complex method



top_image = "ImageExtractorOutput/HouseHD.PNG"

#bottom_image = "./ImageExtractorOutput/IMG_0865.jpeg"
top_depth_map = Depth_Map.generate_depth_map(top_image)

top_scale_factor = Depth_Map.get_simple_scale_factor(int(depth_values[0]))
top_scale_factor_2 = Depth_Map.get_scale_factor(top_depth_map, int(ref_obj_points[0][0]),int(ref_obj_points[0][1]),int(ref_obj_points[1][0]),int(ref_obj_points[1][1]), 400, 400, 720, 942)
print(top_scale_factor)
print(top_scale_factor_2)

exit()
#bottom_scale_factor = 

################################################################
#generate the image masks

#top_mask = Segmentation_Mask.infer_and_create_mask(top_image)#for some reason depthAnything stuffs up CUDA so that YOLO cant use it, so we need to run yolo first (or use CPU)

#bottom_mask = Segmentation_Mask.infer_image(bottom_image)

################################################################
#generate the depth maps
#top_depth_map = Depth_Map.generate_depth_map(top_image)

#bottom_depth_map = Depth_Map.generate_depth_map(bottom_image)

################################################################
#read scale factor
#top_scale = 
#bottom_scale =



################################################################
#convert them to a point cloud

test_point_cloud = Point_Cloud.testing_point_cloud()
test_point_cloud_2 = Point_Cloud.testing_point_cloud()
Point_Cloud.translate_point_cloud(test_point_cloud_2,0,0,10)#dx, dy, dz
Point_Cloud.rotate_point_cloud_X(test_point_cloud,90)


#point_cloud = Point_Cloud.generate_basic_coloured_point_cloud(top_image, top_depth_map, 400, 400)
#point_cloud = Point_Cloud.generate_point_cloud_with_mask(top_image, top_depth_map, FOCAL_LENGTH_X, FOCAL_LENGTH_Y, top_mask, top_scale_factor)#place holder for now in future read scale factor and use mask


################################################################
#merge the two point clouds using translation matrices
final_point_cloud = Point_Cloud.merge_point_cloud(test_point_cloud, test_point_cloud_2)
Point_Cloud.render_point_cloud(final_point_cloud)
################################################################
#reproject and generate new masks

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