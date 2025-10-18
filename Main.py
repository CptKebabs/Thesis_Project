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

# ref_obj_points, depth_values = Depth_Map.read_scale_file("ImageExtractorOutput/BotPers_274_750.scale")#reads as string
# print(ref_obj_points)
# print(depth_values)
#now test the more complex method



top_image_path = "ImageExtractorOutput/Sawmillers_2_1/BotPers_75.png"
top_image_path = "House.PNG"

#bottom_image = "./ImageExtractorOutput/IMG_0865.jpeg"
# top_depth_map = Depth_Map.generate_depth_map(top_image)

# top_scale_factor = Depth_Map.get_simple_scale_factor(int(depth_values[0]))
# top_scale_factor_2 = Depth_Map.get_scale_factor(top_depth_map, int(ref_obj_points[0][0]),int(ref_obj_points[0][1]),int(ref_obj_points[1][0]),int(ref_obj_points[1][1]), 400, 400, 720, 942)
# print(top_scale_factor)
# print(top_scale_factor_2)

# exit()
#bottom_scale_factor = 

################################################################
#generate the image masks

#top_mask = Segmentation_Mask.infer_and_create_mask(top_image)#for some reason depthAnything stuffs up CUDA so that YOLO cant use it, so we need to run yolo first (or use CPU)

#bottom_mask = Segmentation_Mask.infer_image(bottom_image)

################################################################
#generate the depth maps

#top_depth_map = Depth_Map.generate_metric_depth_map(top_image)
top_depth_map = Depth_Map.generate_relative_depth_map_depthanythingV2(top_image_path)

#bottom_depth_map = Depth_Map.generate_depth_map(bottom_image)
pcd = Point_Cloud.generate_basic_coloured_point_cloud(top_image_path,top_depth_map,800,800)

# add_pcd =  Point_Cloud.point_cloud_square(10,(1.0,0,0))
# Point_Cloud.translate_point_cloud(add_pcd,0,0,1000)
# pcd = Point_Cloud.merge_point_cloud(pcd,add_pcd)
#Point_Cloud.correct_relative_point_cloud(pcd)
pcd = pcd.voxel_down_sample(0.01)
Point_Cloud.render_point_cloud(pcd)

top_image = Image.open(top_image_path).convert('RGB')
top_image_width, top_image_height = top_image.size

#intrinsic
top_image_cx = top_image_width / 2
top_image_cy = top_image_height / 2

top_image_fx = 800
top_image_fy = 800

top_image_instrinsic = np.array([[top_image_fx, 0, top_image_cx],
                                 [0, top_image_fy, top_image_cy],
                                 [0, 0, 1]]) # Intrinsic matrix

#extrinsic
top_image_T_extrinsic = np.array([[0.0],[100.0],[0.0]]) # translation
top_image_R_extrinsic = np.array([[0.5],[0.0],[0.0]])  # Rotation all zeros for now

top_image_reconstructed = Point_Cloud.reproject_point_cloud_to_2d_image(pcd,top_image_height,top_image_width,top_image_instrinsic,top_image_T_extrinsic,top_image_R_extrinsic)

import matplotlib.pyplot as plt
plt.imshow(top_image_reconstructed)
plt.show()

    ################################################################
    #convert them to a point cloud

    # test_point_cloud = Point_Cloud.testing_point_cloud()
    # test_point_cloud_2 = Point_Cloud.testing_point_cloud()
    # Point_Cloud.translate_point_cloud(test_point_cloud_2,0,0,10)#dx, dy, dz
    # Point_Cloud.rotate_point_cloud_X(test_point_cloud,90)


# point_cloud = Point_Cloud.generate_basic_coloured_point_cloud(top_image, top_depth_map, 800, 800)

# Point_Cloud.render_point_cloud(point_cloud)

# final = Point_Cloud.point_cloud_axis(1)
# translation_test_x = Point_Cloud.point_cloud_axis(0.1)
# translation_test_y = Point_Cloud.point_cloud_axis(0.1)
# translation_test_z = Point_Cloud.point_cloud_axis(0.1)


# Point_Cloud.translate_point_cloud(translation_test_x,100,0,0)
# Point_Cloud.rotate_point_cloud_X(translation_test_x,90)

# Point_Cloud.translate_point_cloud(translation_test_y,0,100,0)
# Point_Cloud.rotate_point_cloud_Y(translation_test_y,90)


# Point_Cloud.translate_point_cloud(translation_test_z,0,0,100)
# Point_Cloud.rotate_point_cloud_Z(translation_test_z,90)


# Point_Cloud.render_point_cloud(translation_test_x + translation_test_y + translation_test_z +final)




#Point_Cloud.generate_basic_coloured_point_cloud(top_image, top_depth_map, 800, 800)
##Y_axis = Point_Cloud.point_cloud_line(1,(0,1.0,0))
##Z_axis = Point_Cloud.point_cloud_line(1,(0,0,1.0))
#point_cloud_line3 = Point_Cloud.point_cloud_line(1)
#Point_Cloud.translate_point_cloud(point_cloud2,0,0,0)
##Point_Cloud.rotate_point_cloud_X(point_cloud_line2, 90)
#Point_Cloud.rotate_point_cloud_X(point_cloud_line3, 180)

#final = Point_Cloud.merge_point_cloud(final,point_cloud_line3)

#point_cloud = Point_Cloud.generate_point_cloud_with_mask(top_image, top_depth_map, FOCAL_LENGTH_X, FOCAL_LENGTH_Y, top_mask, top_scale_factor)#place holder for now in future read scale factor and use mask


################################################################
#merge the two point clouds using translation matrices
# final_point_cloud = Point_Cloud.merge_point_cloud(test_point_cloud, test_point_cloud_2)
# Point_Cloud.render_point_cloud(final_point_cloud)
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