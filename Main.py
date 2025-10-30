# Main program, will utilise the other subsystems together
import shutil
import sys
import os
import cv2      #computer vision library
import matplotlib.pyplot as plt
import glob
import numpy as np
from PIL import Image
import open3d as o3d #point cloud stuff
#import torch    #Neural network stuff
import subprocess
#This repo imports
import Inference
import bm3d
#YOLO PATHES
#TOP
#
#
#
#BOT
#
#
#
#DEPTH MODELS
#0 = DepthAnything
#1 = DepthPro
#2 = 
img = cv2.imread("./Input/BotPers_S2_2_25834.png")
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = img
img_rgb = img_rgb / 255.0  # Normalize to [0, 1]

#denoised_rgb = cv2.fastNlMeansDenoisingColored(img_rgb, None, 10, 10, 7, 21)

# Set estimated noise level
sigma = 0.05  # try values between 0.02 and 0.1 depending on noise strength

# Denoise using BM3D (color-aware)
denoised_rgb = bm3d.bm3d_rgb(img_rgb, sigma_psd=sigma)
denoised_uint8 = (np.clip(denoised_rgb, 0, 1) * 255).astype(np.uint8)
plt.imshow(denoised_uint8)
plt.show()

cv2.imwrite("BM3D.png",denoised_uint8)
exit()
paths = glob.glob("./Input/*Top*.txt")#get all the top ground truth files

for path in paths:#do this once we have the trained model and know all the file formats so we can test it properly
    print(path)
    top_image_path = ""#add ./input to these
    bot_image_path = ""
    bot_scale_path = ""
    ground_truth_path = ""
    #if all of these exist then
    #run Inference.infer() (we might need to do subprocess because of the weird cuda issue)
    #save outputs IoUs and masks to output folder

#derive mAP from IoU values

#Assumptions:
    #All image, scale and ground truths are to be in the same directory
    #Top image files are named: TopPers_S2_(1-3)_(FrameCount).png
    #Bot image files are named: BotPers_S2_(1-3)_(FrameCount).png
    #Top scale files are named: TopPers_S2_(1-3)_(FrameCount).scale (NOT NEEDED BECAUSE WE CANNOT DERIVE PROPER DEPTH FROM TOP PERS ANYWAY)
    #Bot scale files are named: BotPers_S2_(1-3)_(FrameCount).scale
    #ground truth are (hashval)-TopPers_S2_(1-3)_(FrameCount).txt


    #glob all topPers files and filter them to make sure the other corresponding files are present
    #loop over every topPers that passes this filter
    #running through our code and extracting the orig IoU and our IoU
    #write all results to some file

    #repeat this process for each depth estimator (3) and each YOLO model (3) = 9 times total

result = Inference.infer("TopPers_S2_2_25834.png","BotPers_S2_2_25834.png","TopPers_S2_2_25834.scale","4a3a33f7-BotPers_S2_2_7850.txt",0,"")

print(result[0:2])

import matplotlib.pyplot as plt
plt.imshow(result[2], cmap='gray')  # for binary masks
plt.title('Segmentation Mask')
plt.axis('off')
plt.show()
