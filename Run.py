import numpy
import os
import cv2
import open3d

import Depth_Map
import Point_Cloud

#read paths static for now
top_image_path = "ImageExtractorOutput\Test Pairs\TopPers167.png"
bot_image_path = "ImageExtractorOutput\Test Pairs\BotPers167.png"

top_image_dimensions = cv2.imread(top_image_path).shape #[0] is height [1] is width [2] is rgb channels
bot_image_dimensions = cv2.imread(bot_image_path).shape

print(top_image_dimensions)
print(bot_image_dimensions)

#init camera intrinsics and extrinsics
top_fx = 1.17482757e+03
top_fy = 1.17382323e+03
top_cx = top_image_dimensions[1] / 2.0
top_cy = top_image_dimensions[0] / 2.0

bot_fx = 1.17482757e+03
bot_fy = 1.17382323e+03
bot_cx = bot_image_dimensions[1] / 2.0
bot_cy = bot_image_dimensions[0] / 2.0

top_camera_intrinsic = numpy.array([[[top_fx],  [0],        [top_cx]],
                                    [[0],       [top_fy],   [top_cy]],
                                    [[0],       [0],        [1]]])

top_camera_intrinsic = numpy.array([[[bot_fx],  [0],        [bot_cx]],
                                    [[0],       [bot_fy],   [bot_cy]],
                                    [[0],       [0],        [1]]])

top_camera_T_extrinsic = numpy.array([[0.0],[0.0],[0.0]])#all zeros for now
top_camera_R_extrinsic = numpy.array([[0.0],[0.0],[0.0]])
bot_camera_T_extrinsic = numpy.array([[0.0],[0.0],[0.0]])
bot_camera_R_extrinsic = numpy.array([[0.0],[0.0],[0.0]])

#read scale data
top_scale_path = f"DepthImages\{os.path.basename(os.path.splitext(top_image_path)[0])}_{Depth_Map.INPUT_SIZE}.scale"
bot_scale_path = f"DepthImages\{os.path.basename(os.path.splitext(bot_image_path)[0])}_{Depth_Map.INPUT_SIZE}.scale"

top_ref_obj_points, top_depth_scale_values = Depth_Map.read_scale_file(top_scale_path)#reads as string
bot_ref_obj_points, bot_depth_scale_values = Depth_Map.read_scale_file(bot_scale_path)#reads as string

print(f"Top Scale Points: {top_ref_obj_points}")
print(f"Top Scale Values: {top_depth_scale_values}")
print(f"Bot Scale Points: {bot_ref_obj_points}")
print(f"Bot Scale Values: {bot_depth_scale_values}")
#generate segmentation masks

#generate depth maps
top_depth_map = Depth_Map.generate_metric_depth_map_depthanythingV2(top_image_path,outdoor=False)
bot_depth_map = Depth_Map.generate_metric_depth_map_depthanythingV2(bot_image_path,outdoor=False)

top_depth_map = cv2.bilateralFilter(top_depth_map.astype(numpy.float32), d=5, sigmaColor=0.1, sigmaSpace=5)
bot_depth_map = cv2.bilateralFilter(bot_depth_map.astype(numpy.float32), d=5, sigmaColor=0.1, sigmaSpace=5)

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

# top_scale_factor = Depth_Map.get_simple_scale_factor(top_depth_map,int(top_ref_obj_points[0][0]),int(top_ref_obj_points[0][1]))
# bot_scale_factor = Depth_Map.get_simple_scale_factor(bot_depth_map,int(top_ref_obj_points[1][0]),int(top_ref_obj_points[1][1]))

print(f"Top Scale Factor: {top_scale_factor} m/value")
print(f"Bot Scale Factor: {bot_scale_factor} m/value")

#generate image masks
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
#generate point clouds

top_pcd = Point_Cloud.generate_point_cloud_with_mask(top_image_path,top_depth_map,top_fx,top_fy,top_scale_factor,top_image_mask)
top_cam = Point_Cloud.point_cloud_axis(0.001)


top_pcd = top_pcd + top_cam

#top_pcd, _ = top_pcd.remove_radius_outlier(nb_points=16, radius=0.05) #this is insanely slow

Point_Cloud.rotate_point_cloud_X(top_pcd,-90)#for some reason TRS isnt a thing here. If i do translation first rotates the translation also... probably doing something wrong

Point_Cloud.translate_point_cloud(top_pcd,.040,-.8675,1.2125)#these values vary might derive from extrinsic above later
top_pcd.scale(4.0,center=top_pcd.get_center())

# top_pcd_bbox = top_pcd.get_axis_aligned_bounding_box()
# print("Top Extent:", top_pcd_bbox.get_extent())  # Width, height, depth

bot_ref_obj = Point_Cloud.point_cloud_axis(0.001)
Point_Cloud.translate_point_cloud(bot_ref_obj,0,-.1225,.362)
#rotate and translate in metres and 90d degrees down

bot_pcd = Point_Cloud.generate_point_cloud_with_mask(bot_image_path,bot_depth_map,bot_fx,bot_fy,bot_scale_factor,bot_image_mask)
bot_cam = Point_Cloud.point_cloud_axis(0.001)

# bot_pcd_bbox = bot_pcd.get_axis_aligned_bounding_box()
# print("Bot Extent:", bot_pcd_bbox.get_extent())  # Width, height, depth

bot_pcd = bot_pcd + bot_cam
bot_pcd = bot_pcd + top_cam
bot_pcd = bot_pcd + bot_ref_obj

bot_pcd = bot_pcd + top_pcd
#bot stays same

# top_pcd.transform([#180 deg x and y rot #only for visualisation reprojection comes out normal
#     [1,  0,  0, 0],
#     [0, -1,  0, 0],
#     [0,  0, -1, 0],
#     [0,  0,  0, 1]
# ])


######################
# Point_Cloud.render_point_cloud(top_pcd)

Point_Cloud.render_point_cloud(bot_pcd)


















#merge point clouds



#TODO 
#fill in extrinsic and instrinsic