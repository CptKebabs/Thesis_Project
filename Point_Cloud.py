#use depthAnythingV2 to generate the depth maps, and convert them to a scaled point cloud here
#https://www.youtube.com/watch?v=vGr8Bg2Fda8
import cv2
import open3d
import numpy
from PIL import Image
#call depthAnythingV2 model on image

def generate_basic_coloured_point_cloud(image_path, depth_map, focal_x, focal_y):#based on Depth-Anything-V2s depth_to_pointcloud file
    color_image = cv2.imread(image_path)
    color_image = Image.open(image_path).convert('RGB')
    width, height = color_image.size

    resized_pred = Image.fromarray(depth_map).resize((width, height), Image.NEAREST)#should be the same shape anyway

    x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))#make x and y coordinates corresponding to every possible pixel
    x = (x - width / 2) / focal_x     #need to calibrate and get actual values: focal_length_x
    y = (y - height / 2) / focal_y    #(x-width/2) - set coordinates so that middle becomes 0 - / focal_length_X normalise in terms of camera focal length
    z = numpy.array(resized_pred)# make a new copy of depthmap

    points = numpy.stack((numpy.multiply(x, z), numpy.multiply(y, z), z), axis=-1).reshape(-1, 3)# make an array of [[x0,y0,z0],[x1,y1,z1], ... ,[xn,yn,zn]] where n is pixel num (x and y flattened)
    colors = numpy.array(color_image).reshape(-1, 3) / 255.0   #change rgbs from 0-255 to 0-1 (flatten x and y here too)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    pcd.transform([[-1,0,0,0],
                [0,1,0,0],
                [0,0,-1,0],
                [0,0,0,-1]])#use this for transformations
    return pcd

def generate_point_cloud_with_mask(image_path, depth_map, focal_x, focal_y, mask):
    flat_mask = mask.flatten()
    color_image = cv2.imread(image_path)
    color_image = Image.open(image_path).convert('RGB')
    width, height = color_image.size

    resized_pred = Image.fromarray(depth_map).resize((width, height), Image.NEAREST)#should be the same shape anyway

    x, y = numpy.meshgrid(numpy.arange(width), numpy.arange(height))#make x and y coordinates corresponding to every possible pixel
    x = (x - width / 2) / focal_x     #need to calibrate and get actual values: focal_length_x
    y = (y - height / 2) / focal_y    #(x-width/2) - set coordinates so that middle becomes 0 - / focal_length_X normalise in terms of camera focal length
    z = numpy.array(resized_pred)# make a new copy of depthmap

    points = numpy.stack((numpy.multiply(x, z), numpy.multiply(y, z), z), axis=-1).reshape(-1, 3)# make an array of [[x0,y0,z0],[x1,y1,z1], ... ,[xn,yn,zn]] where n is pixel num (x and y flattened)
    colors = numpy.array(color_image).reshape(-1, 3) / 255.0   #change rgbs from 0-255 to 0-1 (flatten x and y here too)

    #apply mask here
    points = points[flat_mask]
    colors = colors[flat_mask]

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors)

    pcd.transform([[-1,0,0,0],
                [0,1,0,0],
                [0,0,-1,0],
                [0,0,0,-1]])#use this for transformations

    return pcd

def generate_point_clouds(image_path, mask, scale_factor, focal_x, focal_y):
    
    return # the point cloud
    
def render_point_cloud(pcd):
    open3d.visualization.draw_geometries([pcd])




#output scaled point cloud