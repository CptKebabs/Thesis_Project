import numpy
import matplotlib.pyplot as plt
import os
import cv2
import open3d

import Depth_Map
import Point_Cloud
import Segmentation_Mask

DEBUG = True

def infer(top_image_path, bot_image_path, bot_scale_path, top_ground_truth_path, depth_model, yolo_model_path):

    top_image_data = cv2.imread(top_image_path)
    bot_image_data = cv2.imread(bot_image_path)

    top_image_dimensions = top_image_data.shape #[0] is height [1] is width [2] is rgb channels
    bot_image_dimensions = bot_image_data.shape

    #init camera intrinsics and extrinsics
    #TODO IS THE TOP CAMERA INSTRINSIC CORRECT CONSIDERING ITS LOOKING THROUGH WATER? (refraction on water surface causes complexities for focal length) CAN WE IMPROVE IT
    top_fx = 1.17482757e+03 #above water intrinsics
    top_fy = 1.17382323e+03
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
    # top_image_mask = Segmentation_Mask.infer_and_create_mask(top_image_path, yolo_model_path, "Seaweed")
    # bot_image_mask = Segmentation_Mask.infer_and_create_mask(bot_image_path, yolo_model_path, "Seaweed")
    top_image_mask = numpy.zeros((top_image_dimensions[0],top_image_dimensions[1]), dtype=bool)#for now
    bot_image_mask = numpy.zeros((bot_image_dimensions[0],bot_image_dimensions[1]), dtype=bool)

    w = top_image_dimensions[1]
    h = top_image_dimensions[0]

    top_image_mask[h//8 : 7*h//8, w//8 : 7*w//8] = 1

    w = bot_image_dimensions[1]
    h = bot_image_dimensions[0]

    bot_image_mask[h//4 : 3*h//4, w//4 : 3*w//4] = 1


    if depth_model == 0:
        bot_depth_map = Depth_Map.generate_metric_depth_map_depthanythingV2(bot_image_path, outdoor=True)
    elif depth_model == 1:
        bot_depth_map = Depth_Map.generate_metric_depth_map_depthpro(bot_image_path, bot_fx)
    # elif depth_model == 2:
        #bot_depth_map = Depth_Map.

    bot_scale_factor = Depth_Map.get_scale_factor(bot_depth_map, 
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
    top_image_reconstructed_mask = Segmentation_Mask.fill_gaps_in_mask(top_image_reconstructed_mask,3,1)#use a morphological close to fill gaps
    final_top_mask = Segmentation_Mask.add_masks(top_image_mask,top_image_reconstructed_mask)#combined orig predicted mask with the pointcloud mask from bottom perspective
    
    top_ground_truth = Segmentation_Mask.yolo_to_numpy(top_ground_truth_path, top_image_dimensions)#Read yolo format ground truth mask

    original_IoU = Segmentation_Mask.calculate_IoU(top_ground_truth, top_image_mask)#calculate IoU from original mask and groundtruth
    reconstructed_IoU = Segmentation_Mask.calculate_IoU(top_ground_truth,final_top_mask)#calculate IoU from our final mask and groundtruth 

    return [original_IoU, reconstructed_IoU, final_top_mask]