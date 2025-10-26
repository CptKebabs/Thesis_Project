import numpy
import os
import cv2
import open3d

import Depth_Map
import Point_Cloud
import Segmentation_Mask

#read paths static for now (either arg input or Glob loop in future)
# top_image_path = "ImageExtractorOutput\Test Pairs\TopPers167.png"
# bot_image_path = "ImageExtractorOutput\Test Pairs\BotPers167.png"
# top_image_path = "ImageExtractorOutput\Test Pairs\TopPers612.png"
# bot_image_path = "ImageExtractorOutput\Test Pairs\BotPers612.png"
top_image_path = "TopPers_S2_2_25834.png"
bot_image_path = "BotPers_S2_2_25834.png"

top_image_dimensions = cv2.imread(top_image_path).shape #[0] is height [1] is width [2] is rgb channels
bot_image_dimensions = cv2.imread(bot_image_path).shape

print(top_image_dimensions)
print(bot_image_dimensions)

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

#top_camera_T_extrinsic = numpy.array([[-.040],[.8675],[-1.2125]])#original translation vector using instead the 90 degree rotation of this
#these need to be the inverse of the point cloud translation and rotations
#top_camera_R_extrinsic = numpy.array([[-numpy.pi/2.0],[0.0],[0.0]])#original rotation vector using instead the negation of this
#(because TRS the rotation is applied after the translation in the reprojection, which causes the whole thing to rotate.
#So i just got the translation and rotation that works with that)

top_camera_T_extrinsic = numpy.array([[-.040],[1.2125],[.8675]])
top_camera_R_extrinsic = numpy.array([[numpy.pi/2.0],[0.0],[0.0]])

bot_camera_T_extrinsic = numpy.array([[0.0],[0.0],[0.0]])
bot_camera_R_extrinsic = numpy.array([[0.0],[0.0],[0.0]])

#read scale data
# top_scale_path = f"DepthImages\{os.path.basename(os.path.splitext(top_image_path)[0])}_{Depth_Map.INPUT_SIZE}.scale"
# bot_scale_path = f"DepthImages\{os.path.basename(os.path.splitext(bot_image_path)[0])}_{Depth_Map.INPUT_SIZE}.scale"
top_scale_path = f"{os.path.splitext(top_image_path)[0]}.scale"
bot_scale_path = f"{os.path.splitext(bot_image_path)[0]}.scale"

top_ref_obj_points, top_depth_scale_values = Depth_Map.read_scale_file(top_scale_path)#reads as string
bot_ref_obj_points, bot_depth_scale_values = Depth_Map.read_scale_file(bot_scale_path)#reads as string

print(f"Top Scale Points: {top_ref_obj_points}")
print(f"Bot Scale Points: {bot_ref_obj_points}")

#generate segmentation masks
#TODO make this YOLO once we have model

top_image_mask = numpy.zeros((top_image_dimensions[0],top_image_dimensions[1]), dtype=bool)#for now
bot_image_mask = numpy.zeros((bot_image_dimensions[0],bot_image_dimensions[1]), dtype=bool)

w = top_image_dimensions[1]
h = top_image_dimensions[0]

top_image_mask[h//8 : 7*h//8, w//8 : 7*w//8] = 1

import matplotlib.pyplot as plt

plt.imshow(top_image_mask, cmap='gray')  # for binary masks
plt.title('Segmentation Mask')
plt.axis('off')
plt.show()

w = bot_image_dimensions[1]
h = bot_image_dimensions[0]

bot_image_mask[h//4 : 3*h//4, w//4 : 3*w//4] = 1

plt.imshow(bot_image_mask, cmap='gray')  # for binary masks
plt.title('Segmentation Mask')
plt.axis('off')
plt.show()

#generate depth maps
top_depth_map = Depth_Map.generate_metric_depth_map_depthanythingV2(top_image_path, outdoor=True)
bot_depth_map = Depth_Map.generate_metric_depth_map_depthanythingV2(bot_image_path, outdoor=True)

# top_depth_map = Depth_Map.generate_metric_depth_map_depthpro(top_image_path, top_fx)
# bot_depth_map = Depth_Map.generate_metric_depth_map_depthpro(bot_image_path, bot_fx)

# top_depth_map = cv2.bilateralFilter(top_depth_map.astype(numpy.float32), d=5, sigmaColor=0.1, sigmaSpace=5)#not really needed but helps smooth for now
# bot_depth_map = cv2.bilateralFilter(bot_depth_map.astype(numpy.float32), d=5, sigmaColor=0.1, sigmaSpace=5)

top_scale_factor = Depth_Map.get_scale_factor(top_depth_map, 
                                              int(top_ref_obj_points[0][0]),
                                              int(top_ref_obj_points[0][1]),
                                              int(top_ref_obj_points[1][0]),
                                              int(top_ref_obj_points[1][1]),
                                              top_fx,  
                                              top_fy, 
                                              top_cx, 
                                              top_cy)

bot_scale_factor = Depth_Map.get_scale_factor(bot_depth_map, 
                                              int(bot_ref_obj_points[0][0]),
                                              int(bot_ref_obj_points[0][1]),
                                              int(bot_ref_obj_points[1][0]),
                                              int(bot_ref_obj_points[1][1]),
                                              bot_fx,  
                                              bot_fy, 
                                              bot_cx, 
                                              bot_cy)

print(f"Top Scale Factor: {top_scale_factor} m/value")
print(f"Bot Scale Factor: {bot_scale_factor} m/value")

#generate point clouds

#top
top_pcd = Point_Cloud.generate_point_cloud_with_mask(top_image_path,top_depth_map,top_fx,top_fy,top_scale_factor,top_image_mask)#main cloud
top_cam = Point_Cloud.point_cloud_axis(0.001)#camera
top_ref_obj = Point_Cloud.point_cloud_axis(0.0005)#ref obj
Point_Cloud.translate_point_cloud(top_ref_obj,0.0,.1225,.362)

top_pcd = top_pcd + top_cam #merge
top_pcd = top_pcd + top_ref_obj

#try move the point cloud and match with the camera extrinsic
Point_Cloud.translate_point_cloud(top_pcd,-top_camera_T_extrinsic[0],-top_camera_T_extrinsic[1],-top_camera_T_extrinsic[2])#translations need to be inversion of extrinsic or vice versa
Point_Cloud.rotate_point_cloud_X(top_pcd,numpy.rad2deg(-top_camera_R_extrinsic[0]))

Point_Cloud.render_point_cloud(top_pcd)



#bot
bot_ref_obj = Point_Cloud.point_cloud_axis(0.001)
Point_Cloud.translate_point_cloud(bot_ref_obj,0.0,-.1225,.362)

bot_pcd = Point_Cloud.generate_point_cloud_with_mask(bot_image_path,bot_depth_map,bot_fx,bot_fy,bot_scale_factor,bot_image_mask)
bot_cam = Point_Cloud.point_cloud_axis(0.001)

bot_pcd = bot_pcd + bot_cam
bot_pcd = bot_pcd + bot_ref_obj

Point_Cloud.render_point_cloud(bot_pcd)

final_pcd = bot_pcd
Point_Cloud.render_point_cloud(final_pcd)

top_image_reconstructed = Point_Cloud.reproject_point_cloud_to_2d_image(final_pcd,top_image_dimensions[0],top_image_dimensions[1],top_camera_intrinsic,top_camera_T_extrinsic,top_camera_R_extrinsic)
bot_image_reconstructed = Point_Cloud.reproject_point_cloud_to_2d_image(final_pcd,bot_image_dimensions[0],bot_image_dimensions[1],bot_camera_intrinsic,bot_camera_T_extrinsic,bot_camera_R_extrinsic)

#TODO need to convert from sparse point cloud to dense map here to make the mask not have gaps
# alpha = 0.03
# # Compute the mesh
# mesh = open3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(final_pcd, alpha) This takes way too long 
# # Optionally, compute vertex normals for shading
# #mesh.compute_vertex_normals()
# # Visualize
# open3d.visualization.draw_geometries([mesh])




#raw image reprojections
import matplotlib.pyplot as plt
plt.imshow(top_image_reconstructed)
plt.show()

plt.imshow(bot_image_reconstructed)
plt.show()

#raw image convented to a boolean array (mask)
top_image_reconstructed_mask = Segmentation_Mask.reprojection_to_mask(top_image_reconstructed)#set pixels that are coloured to True

plt.imshow(top_image_reconstructed_mask, cmap='gray')
plt.title("Boolean Mask")
plt.axis('off')
plt.show()

#gaps filled
top_image_reconstructed_mask = Segmentation_Mask.fill_gaps_in_mask(top_image_reconstructed_mask,3)#use a morphological close

plt.imshow(top_image_reconstructed_mask, cmap='gray')
plt.title("Boolean Mask")
plt.axis('off')
plt.show()

#now add it to the top YOLO only prediction mask
final_top_mask = Segmentation_Mask.add_masks(top_image_mask,top_image_reconstructed_mask)#combined orig predicted mask with the pointcloud mask from bottom perspective

plt.imshow(final_top_mask, cmap='gray')
plt.title("Boolean Mask")
plt.axis('off')
plt.show()


#read ground truth
test_mask = Segmentation_Mask.yolo_to_numpy("4a3a33f7-BotPers_S2_2_7850.txt", bot_image_dimensions)#Read yolo format

plt.imshow(test_mask, cmap='gray')
plt.title("Boolean Mask")
plt.axis('off')
plt.show()

print(test_mask.shape)

#compare with IoU to ground truth and get IoU of new prediction
print(Segmentation_Mask.calculate_IoU(top_image_mask,bot_image_mask))#calculate IoU from mask and groundtruth later











#top_pcd.scale(4.0,center=top_pcd.get_center())

# top_pcd_bbox = top_pcd.get_axis_aligned_bounding_box()
# print("Top Extent:", top_pcd_bbox.get_extent())  # Width, height, depth