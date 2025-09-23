# Main program, will utilise the other subsystems together
import sys
import os
import cv2      #computer vision library
import matplotlib
import numpy as np
from PIL import Image
import open3d as o3d #point cloud stuff
import torch    #Neural network stuff

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

sys.path.append(os.path.abspath('../Depth-Anything-V2'))

from depth_anything_v2.dpt import DepthAnythingV2

#from DepthAnythingV2 repo:
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'DepthModel/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread(image_path)
depth = model.infer_image(raw_img,500) #can vary the input size as a second parameter here default is probably 518# HxW raw depth map in numpy

print("input image details:")
print(len(raw_img))#image height
print(len(raw_img[0]))#image length
#

print("output depthmap details:")
print(type(depth))#depth is of type numpy.ndarray
print(depth.dtype)#and the ndarray is of type float32
print(depth.ndim)#2d array
print(len(depth))#depthmap height 2098
print(len(depth[0]))#depthmap length

depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 #distribiute values between 0 and 255
depth = depth.astype(np.uint8)#change the array to type uint8 (0-255)

cmap = matplotlib.colormaps.get_cmap('Spectral_r')#spectral reversed 
print(depth)
depthImage = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        #     cmap(depth) - convert the depthMap to a coloured image by passing it through a colour map     
        #                [:, :, :3] - extract rgb only from colourmap output,
                             # * 255 because output is 0-1 range
                                    #[:, :, ::-1] reverse the order because its in BGR
                                                #.astype(np.uint8) convert to int 0-255 instead of floats
print(type(depthImage))

cv2.imwrite(os.path.join("./", os.path.splitext(os.path.basename("IMG_0865Depth500.jpeg"))[0] + '.png'), depthImage)#save this new depth image


###########################################

# color_image = Image.open(image_path).convert('RGB')
# width, height = color_image.size

# resized_pred = Image.fromarray(depth).resize((width, height), Image.NEAREST)

# x, y = np.meshgrid(np.arange(width), np.arange(height))
# x = (x - width / 2) / 200     #need to calibrate and get actual values: focal_length_x
# y = (y - height / 2) / 200    #focal_length_y
# z = np.array(resized_pred)
# points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
# colors = np.array(color_image).reshape(-1, 3) / 255.0

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# pcd.transform([[-1,0,0,0],
#                [0,1,0,0],
#                [0,0,-1,0],
#                [0,0,0,-1]])#use this for transformations

# o3d.visualization.draw_geometries([pcd])#visualise the point cloud


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