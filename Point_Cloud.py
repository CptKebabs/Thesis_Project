#use depthAnythingV2 to generate the depth maps, and convert them to a scaled point cloud here
#https://www.youtube.com/watch?v=vGr8Bg2Fda8
import cv2
import open3d
import numpy
import math
from PIL import Image
#call depthAnythingV2 model on image

def single_point():
    colours = numpy.array([[0,1.0,1.0]])
    points = numpy.array([[0,0,0]])
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colours)
    return pcd

def point_cloud_line_Z(Scale, colour):
    colours = numpy.full((1000,3),colour)
    points = []
    for i in range(1000):
        z = i * (.1) * Scale
        points.append([0,0,z])
    points = numpy.array(points)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colours)
    return pcd

def point_cloud_line_X(Scale, colour):
    colours = numpy.full((1000,3),colour)
    points = []
    for i in range(1000):
        x = i * (.1) * Scale
        points.append([x,0,0])
    points = numpy.array(points)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colours)
    return pcd

def point_cloud_line_Y(Scale, colour):
    colours = numpy.full((1000,3),colour)
    points = []
    for i in range(1000):
        y = i * (.1) * Scale
        points.append([0,y,0])
    points = numpy.array(points)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colours)
    return pcd

def point_cloud_axis(Scale):
    origin = single_point()
    X_axis = point_cloud_line_X(Scale,(1.0,0,0))
    Y_axis = point_cloud_line_Y(Scale,(0,1.0,0))
    Z_axis = point_cloud_line_Z(Scale,(0,0,1.0))
    final = origin + X_axis
    final = final + Y_axis
    final = final + Z_axis
    return final


def point_cloud_square(Scale, Colour):
    colours = numpy.full((10000,3),Colour)
    points = []
    for i in range(100):
        for j in range(100):
            x = j * (.1) * Scale
            y = i * (.1) * Scale
            points.append([x,y,0])
    points = numpy.array(points)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colours)
    return pcd

def generate_basic_coloured_point_cloud(image_path, depth_map, focal_x, focal_y):#based on Depth-Anything-V2s depth_to_pointcloud file

    color_image = Image.open(image_path).convert('RGB')
    width, height = color_image.size

    resized_pred = Image.fromarray(depth_map).resize((width, height), Image.NEAREST)#should be the same shape anyway

    x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))#make x and y coordinates corresponding to every possible pixel
    x = (x - width / 2) / focal_x     #need to calibrate and get actual values: focal_length_x and cx cy 
    y = (y - height / 2) / focal_y    #(x-width/2) - set coordinates so that middle becomes 0 - / focal_length_X normalise in terms of camera focal length
    z = numpy.array(resized_pred)# make a new copy of depthmap

    points = numpy.stack((numpy.multiply(x, z), numpy.multiply(y, z), z), axis=-1).reshape(-1, 3)# make an array of [[x0,y0,z0],[x1,y1,z1], ... ,[xn,yn,zn]] where n is pixel num (x and y flattened)
    colors = numpy.array(color_image).reshape(-1, 3) / 255.0   #change rgbs from 0-255 to 0-1 (flatten x and y here too)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd

def generate_basic_coloured_point_cloud_with_scale(image_path, depth_map, focal_x, focal_y, scale_factor):#based on Depth-Anything-V2s depth_to_pointcloud file
    color_image = Image.open(image_path).convert('RGB')
    width, height = color_image.size

    resized_pred = Image.fromarray(depth_map).resize((width, height), Image.NEAREST)#should be the same shape anyway

    x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))#make x and y coordinates corresponding to every possible pixel
    x = (x - width / 2) / focal_x     #need to calibrate and get actual values: focal_length_x
    y = (y - height / 2) / focal_y    #(x-width/2) - set coordinates so that middle becomes 0 - / focal_length_X normalise in terms of camera focal length
    z = numpy.array(resized_pred)# make a new copy of depthmap

    points = numpy.stack((numpy.multiply(x, z), numpy.multiply(y, z), z), axis=-1).reshape(-1, 3) * scale_factor# make an array of [[x0,y0,z0],[x1,y1,z1], ... ,[xn,yn,zn]] where n is pixel num (x and y flattened)
    colors = numpy.array(color_image).reshape(-1, 3) / 255.0   #change rgbs from 0-255 to 0-1 (flatten x and y here too)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd

def generate_point_cloud_with_mask(image_path, depth_map, focal_x, focal_y, scale_factor, mask):
    flat_mask = mask.flatten()
    color_image = Image.open(image_path).convert('RGB')
    width, height = color_image.size

    resized_pred = Image.fromarray(depth_map).resize((width, height), Image.NEAREST)#should be the same shape anyway

    x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))#make x and y coordinates corresponding to every possible pixel
    x = (x - width / 2) / focal_x     #need to calibrate and get actual values: focal_length_x
    y = (y - height / 2) / focal_y    #(x-width/2) - set coordinates so that middle becomes 0 - / focal_length_X normalise in terms of camera focal length
    z = numpy.array(resized_pred)# make a new copy of depthmap

    points = numpy.stack((numpy.multiply(x, z), numpy.multiply(y, z), z), axis=-1).reshape(-1, 3) * scale_factor # make an array of [[x0,y0,z0],[x1,y1,z1], ... ,[xn,yn,zn]] where n is pixel num (x and y flattened)
    colors = numpy.array(color_image).reshape(-1, 3) / 255.0   #change rgbs from 0-255 to 0-1 (flatten x and y here too)

    #apply mask here
    points = points[flat_mask]
    colors = colors[flat_mask]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    return pcd

def generate_point_cloud_with_open3d(image_path, depth_map, focal_x, focal_y):
    image_arr = cv2.imread(image_path)


    width = image_arr.shape[1]
    height = image_arr.shape[0]
    cx = width / 2
    cy = height / 2

    depth_o3d = open3d.geometry.Image(depth_map)

    intrinsics = open3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, focal_x, focal_y, cx, cy)

    pcd = open3d.geometry.PointCloud.create_from_depth_image(#there is a parameter for this that is literally the entire extrinsic matrix may mean we dont even need to do explicit
        depth = depth_o3d,                                #conversions
        intrinsic=intrinsics,
        depth_scale=1000.0,
        depth_trunc=3.0,#supposed to cull points that are beyond the input range
        stride=1,#sampling rate of pixels 1 = every pixel 2 = every 2
        project_valid_depth_only=True
    )
    # color_image = Image.open(image_path).convert('RGB')
    # colours = numpy.array(color_image).reshape(-1, 3).astype(numpy.float32) / 255.0#only works for metric for some reason???
    # pcd.colors = open3d.utility.Vector3dVector(colours)

    return pcd

def generate_coloured_point_cloud_with_open3d(image_path, depth_map, focal_x, focal_y, mask, scale_factor): #works also but is exact same as what we already have
    # # Load RGB image 
    # top_image_bgr = cv2.imread(bot_image_path)
    # top_image_rgb = cv2.cvtColor(top_image_bgr, cv2.COLOR_BGR2RGB)

    # # Convert to Open3D image formats
    # color_o3d = open3d.geometry.Image(top_image_rgb.astype(numpy.uint8))
    # depth_o3d = open3d.geometry.Image(top_depth_map.astype(numpy.float32))

    # intrinsics = open3d.camera.PinholeCameraIntrinsic()
    # intrinsics.set_intrinsics(top_image_bgr.shape[1], top_image_bgr.shape[0], 1.17482757e+03, 1.17382323e+03, top_image_bgr.shape[1]/2, top_image_bgr.shape[0]/2)

    # rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
    #     color_o3d, depth_o3d,
    #     depth_scale=1.0,  # Adjust depending on depth units
    #     depth_trunc=100.0,
    #     convert_rgb_to_intensity=False
    # )

    # # Create point cloud from RGBD and camera intrinsics
    # pcd = open3d.geometry.PointCloud.create_from_rgbd_image(
    #     rgbd_image,
    #     intrinsics
    # )
    return


#remember TRS order
def rotate_point_cloud_X(pcd, deg):#degrees down also anti-clockwise i guess (about the x axis)
    rad = math.radians(deg)
    pcd.transform([ [1,0,0,0],
                    [0,math.cos(rad),-math.sin(rad),0],
                    [0,math.sin(rad),math.cos(rad),0],
                    [0,0,0,1]])
    return pcd

def rotate_point_cloud_Y(pcd, deg):# anti-clockwise (about the y axis)
    rad = math.radians(deg)
    pcd.transform([ [math.cos(rad),0,math.sin(rad),0],
                    [0,1,0,0],
                    [-math.sin(rad),0,math.cos(rad),0],
                    [0,0,0,1]])
    return pcd

def rotate_point_cloud_Z(pcd, deg):#anti-clockwise (about the z axis)
    rad = math.radians(deg)
    pcd.transform([ [math.cos(rad),-math.sin(rad),0,0],
                    [math.sin(rad),math.cos(rad),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
    return pcd

def translate_point_cloud(pcd,dx,dy,dz):#in my case just going to shift the top point cloud in the -z direction
    vector = numpy.array([dx,dy,dz])
    pcd.translate(vector)
    return pcd

def correct_relative_point_cloud(pcd):
    pcd.transform([[-1,0,0,0],# this is just rotate_y 180 degrees but with [3][3] = -1
                   [0,1,0,0],
                   [0,0,-1,0],
                   [0,0,0,-1]])#use this for transformations

def render_point_cloud(pcd):
    open3d.visualization.draw_geometries([pcd])

def merge_point_cloud(pcd_1, pcd_2):
    return pcd_1 + pcd_2

def reproject_point_cloud_to_2d_image(pcd,image_height,image_width,intrinsic,extrinsic_t,extrinsic_r):
    point_cloud_points = numpy.asarray(pcd.points)
    point_cloud_colors = numpy.asarray(pcd.colors)#these are in 0-1 float range
    RGB_point_cloud_colors = (point_cloud_colors * 255).astype(numpy.uint8)#convert them to rgb 0-255 range

    image_points, _ = cv2.projectPoints(point_cloud_points, extrinsic_r, extrinsic_t, intrinsic, distCoeffs=None)# calculate the screen coordinates from the pointcloud
    image_points = image_points.reshape(-1, 2)#change to [x,y],[x,y] format

    image_reconstructed = numpy.zeros((image_height, image_width, 3), dtype=numpy.uint8)#create the new blank image array

    u_points, v_points = image_points[:, 0], image_points[:, 1]# get the x/u and y/v coordinate from each [x,y] pair
    u_int = numpy.clip(numpy.round(u_points).astype(int), 0, image_width - 1)# round and ensure they are within image bounds (may need modification for the other point cloud as clip)
    v_int = numpy.clip(numpy.round(v_points).astype(int), 0, image_height - 1)#may cause the other pixels to start wrapping onto the image edges

    valid_indices = (u_int >= 0) & (v_int >= 0)  # Check if points are within valid bounds not sure if necessary considering the prev step
    image_reconstructed[v_int[valid_indices], u_int[valid_indices]] = RGB_point_cloud_colors[valid_indices]

    return image_reconstructed


def combined_rotation_matrix_xyz(x_deg, y_deg, z_deg):#testing to see if this fixes the issue with reprojection not lining up with translated positions
    x_rad, y_rad, z_rad = numpy.radians([x_deg, y_deg, z_deg])
    
    Rx = numpy.array([
        [1, 0, 0, 0],
        [0, numpy.cos(x_rad), -numpy.sin(x_rad), 0],
        [0, numpy.sin(x_rad),  numpy.cos(x_rad), 0],
        [0, 0, 0, 1]
    ])
    
    Ry = numpy.array([
        [numpy.cos(y_rad), 0, numpy.sin(y_rad), 0],
        [0, 1, 0, 0],
        [-numpy.sin(y_rad), 0, numpy.cos(y_rad), 0],
        [0, 0, 0, 1]
    ])
    
    Rz = numpy.array([
        [numpy.cos(z_rad), -numpy.sin(z_rad), 0, 0],
        [numpy.sin(z_rad),  numpy.cos(z_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    #Matrix multiplication is apparently in reverse order of application
    #So this applies Rx, then Ry, then Rz
    ret_matrix = Rz @ Ry @ Rx
    return ret_matrix

def find_closest_point(pcd, point = numpy.array([0.0,0.0,0.0])):
    pcd_points = numpy.asarray(pcd.points)
    top_distances = numpy.linalg.norm(pcd_points - point, axis=1)
    min_distance_idx = numpy.argmin(top_distances)#we do this bacause we want to know what the z component is not just the distance
    min_distance_point = pcd_points[min_distance_idx]
    return min_distance_point

#output scaled point cloud