import numpy
import matplotlib.pyplot as plt
import os
import cv2
import open3d
import bm3d
from PIL import Image

import Depth_Map
import Point_Cloud
import Segmentation_Mask

DEBUG = True
SIGMA_PSD = 0.05#the amount of noise present in the image

def infer(top_image_path, bot_image_path, bot_scale_path, top_ground_truth_path, depth_model, model_size):

    top_image_cv2 = cv2.imread(top_image_path)#bgr
    bot_image_cv2 = cv2.imread(bot_image_path)

    top_image_dimensions = top_image_cv2.shape#[0] is height [1] is width [2] is rgb channels
    bot_image_dimensions = bot_image_cv2.shape


    print("Running BM3D")#run noise reduction
    top_image_cv2 = top_image_cv2 / 255.0#normalise to 0-1 range for bm3d
    bot_image_cv2 = top_image_cv2 / 255.0#normalise to 0-1 range for bm3d

    top_image_cv2 = bm3d.bm3d_rgb(top_image_cv2, sigma_psd = SIGMA_PSD)#run bm3d on image data
    top_image_cv2 = (numpy.clip(top_image_cv2, 0, 1) * 255).astype(numpy.uint8)#convert back to 0-255 values

    bot_image_cv2 = bm3d.bm3d_rgb(bot_image_cv2, sigma_psd = SIGMA_PSD)
    bot_image_cv2 = (numpy.clip(bot_image_cv2, 0, 1) * 255).astype(numpy.uint8)

    top_image_PIL = cv2.cvtColor(top_image_cv2, cv2.COLOR_BGR2RGB)#convert to PILs RGB format (opencv is BGR)
    top_image_PIL = Image.fromarray(top_image_PIL)

    bot_image_PIL = cv2.cvtColor(bot_image_cv2, cv2.COLOR_BGR2RGB)#convert to PILs RGB format (opencv is BGR)
    bot_image_PIL = Image.fromarray(bot_image_PIL)

    

    #init camera intrinsics and extrinsics
    top_fx = 1.51350708e+03 #above water intrinsics
    top_fy = 1.51350708e+03
    top_cx = top_image_dimensions[1] / 2.0
    top_cy = top_image_dimensions[0] / 2.0

    bot_fx = 1.51350708e+03 #below water intrinsics
    bot_fy = 1.51350708e+03
    bot_cx = bot_image_dimensions[1] / 2.0
    bot_cy = bot_image_dimensions[0] / 2.0

    top_camera_intrinsic = numpy.array([[[top_fx],  [0],        [top_cx]],
                                        [[0],       [top_fy],   [top_cy]],
                                        [[0],       [0],        [1]]])

    bot_camera_intrinsic = numpy.array([[[bot_fx],  [0],        [bot_cx]],
                                        [[0],       [bot_fy],   [bot_cy]],
                                        [[0],       [0],        [1]]])
    
    top_camera_T_extrinsic = numpy.array([[-.040],[1.2125],[.8675]])
    top_camera_R_extrinsic = numpy.array([[numpy.pi/2.0],[0.0],[0.0]])

    bot_camera_T_extrinsic = numpy.array([[0.0],[0.0],[0.0]])#not using this because top pers depth doesnt come out as required
    bot_camera_R_extrinsic = numpy.array([[0.0],[0.0],[0.0]])

    #read scale data
    bot_ref_obj_points, bot_depth_scale_values = Depth_Map.read_scale_file(bot_scale_path)#reads as string

    print(f"Bot Scale Points: {bot_ref_obj_points}")

    #generate segmentation masks
    top_model_path = f"./YoloModels/top_{model_size}.pt"
    bot_model_path = f"./YoloModels/bot_{model_size}.pt"

    top_image_mask = Segmentation_Mask.infer_and_create_mask(top_image_path, top_model_path, "Seaweed")
    bot_image_mask = Segmentation_Mask.infer_and_create_mask(bot_image_path, bot_model_path, "Seaweed")
   
    if depth_model == 0:
        print("Generating Depth with DepthAnything")
        bot_depth_map = Depth_Map.generate_metric_depth_map_depthanythingV2(bot_image_cv2, outdoor=True)
    elif depth_model == 1:
        print("Generating Depth with DepthPro")
        bot_depth_map = Depth_Map.generate_metric_depth_map_depthpro(bot_image_PIL, bot_fx)
    elif depth_model == 2:
        print("Generating Depth with UDepth")
        bot_depth_map = Depth_Map.generate_relative_depthmap_UDepth(bot_image_PIL, bot_image_mask)
        bot_depth_map = cv2.resize(bot_depth_map, (bot_image_dimensions[1], bot_image_dimensions[0]),interpolation = cv2.INTER_NEAREST)

    bot_scale_factor, scale_point1, scale_point2 = Depth_Map.get_scale_factor(bot_depth_map, 
                                              int(bot_ref_obj_points[0][0]),
                                              int(bot_ref_obj_points[0][1]),
                                              int(bot_ref_obj_points[1][0]),
                                              int(bot_ref_obj_points[1][1]),
                                              bot_fx,  
                                              bot_fy, 
                                              bot_cx, 
                                              bot_cy)
    
    print(f"Bot Scale Factor: {bot_scale_factor} m/value")

    final_pcd = Point_Cloud.generate_point_cloud_with_mask(bot_image_path,bot_depth_map,bot_fx,bot_fy,bot_scale_factor,bot_image_mask)
  
    top_image_reconstructed = Point_Cloud.reproject_point_cloud_to_2d_image(final_pcd,top_image_dimensions[0],top_image_dimensions[1],top_camera_intrinsic,top_camera_T_extrinsic,top_camera_R_extrinsic)#look at the bot point cloud from the top perspective

    top_image_reconstructed_mask = Segmentation_Mask.reprojection_to_mask(top_image_reconstructed)#set pixels that are coloured to True
    top_image_reconstructed_mask = Segmentation_Mask.fill_gaps_in_mask(top_image_reconstructed_mask,5,3)#use a morphological close to fill gaps
    final_top_mask = Segmentation_Mask.add_masks(top_image_mask,top_image_reconstructed_mask)#combined orig predicted mask with the pointcloud mask from bottom perspective
    
    top_ground_truth = Segmentation_Mask.yolo_to_numpy(top_ground_truth_path, top_image_dimensions)#Read yolo format ground truth mask

    original_IoU = Segmentation_Mask.calculate_IoU(top_ground_truth, top_image_mask)#calculate IoU from original mask and groundtruth
    reconstructed_IoU = Segmentation_Mask.calculate_IoU(top_ground_truth,final_top_mask)#calculate IoU from our final mask and groundtruth 

    original_dice = Segmentation_Mask.calculate_dice(top_ground_truth, top_image_mask)
    reconstructed_dice = Segmentation_Mask.calculate_dice(top_ground_truth, final_top_mask)

    scale_point1_norm = (scale_point1-bot_depth_map.min())/(bot_depth_map.max()- bot_depth_map.min())
    scale_point2_norm = (scale_point2-bot_depth_map.min())/(bot_depth_map.max()- bot_depth_map.min())

    return [original_IoU, reconstructed_IoU, original_dice, reconstructed_dice, final_top_mask, top_ground_truth, top_image_mask, top_image_reconstructed_mask, bot_image_mask, scale_point1, scale_point2, scale_point1_norm, scale_point2_norm, bot_depth_map.max(), bot_depth_map.min()]