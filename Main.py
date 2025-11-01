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
import Depth_Map
import Point_Cloud
import Segmentation_Mask
# import bm3d

#YOLO PATHES #program just takes input: large med or small
#TOP
# ./YoloModels/top_large
# ./YoloModels/top_med
# ./YoloModels/top_nano
#BOT
# ./YoloModels/bot_large
# ./YoloModels/bot_med
# ./YoloModels/bot_nano
#DEPTH MODELS
#0 = DepthAnything
#1 = DepthPro
#2 = UDepth (relative only) (and has to resize image to 640,480 and mask to half that)

paths = glob.glob("./Input/*Top*.txt")#get all the top ground truth files
count = 1 
original_IoUs = []
reconstructed_IoUs = []
original_dices = []
reconstructed_dices = []

scale_points = []
normalised_scale_points = []
min_max_depths = []

output_dir = "UDepth_N/" 
yolo_model = "nano"
depth_model = 2

for path in paths:#do this once we have the trained model and know all the file formats so we can test it properly
    print(f"Now running {path}")
    base = os.path.splitext(path)[0]#get rid of extension
    base = os.path.basename(base)#get rid of folders
    top_image_path = "./Input/" +base[9:] + ".png"#get rid of has thing on end of ground truth file that label studio adds
    bot_image_path = "./Input/Bot" + base[12:] + ".png"
    bot_scale_path = "./Input/Bot" + base[12:] + ".scale"
    ground_truth_path = path

    if os.path.exists(top_image_path) and os.path.exists(bot_image_path) and os.path.exists(bot_scale_path) and os.path.exists(ground_truth_path):#if all files present
        print(f"Pair {count} of {len(paths)}")
        result = Inference.infer(top_image_path,bot_image_path,bot_scale_path,ground_truth_path,depth_model,yolo_model)
        print(f"Original IoU: {result[0]}")
        print(f"Reconstructed IoU: {result[1]}")
        print(f"Original Dice: {result[2]}")
        print(f"Reconstructed Dice: {result[3]}")
        original_IoUs.append(result[0])
        reconstructed_IoUs.append(result[1])
        original_dices.append(result[2])
        reconstructed_dices.append(result[3])
        Segmentation_Mask.save_mask(f"./Output/{output_dir}{yolo_model}_{depth_model}_{base[17:]}_FinalMask.png",result[4])
        Segmentation_Mask.save_mask(f"./Output/{output_dir}{yolo_model}_{depth_model}_{base[17:]}_GroundTruth.png",result[5])
        Segmentation_Mask.save_mask(f"./Output/{output_dir}{yolo_model}_{depth_model}_{base[17:]}_TopRawPrediction.png",result[6])
        Segmentation_Mask.save_mask(f"./Output/{output_dir}{yolo_model}_{depth_model}_{base[17:]}_BottomReconMask.png",result[7])
        Segmentation_Mask.save_mask(f"./Output/{output_dir}{yolo_model}_{depth_model}_{base[17:]}_BotRawPrediction.png",result[8])
        scale_points.append((result[9],result[10]))
        normalised_scale_points.append((result[11],result[12]))
        min_max_depths.append((result[13],result[14]))
        count += 1

    else:
        print(f"Skipping: {path}, missing files")
    
save_IoU_arr_orig = np.array(original_IoUs)
save_IoU_arr_ours = np.array(reconstructed_IoUs)
save_dice_arr_orig = np.array(original_dices)
save_dice_arr_ours = np.array(reconstructed_dices)

scale_points_arr = np.array(scale_points)
norm_scale_points_arr = np.array(normalised_scale_points)
min_max_depths_arr = np.array(min_max_depths)

np.save(f"./Output/{output_dir}/original_IoUs.npy", save_IoU_arr_orig)
np.save(f"./Output/{output_dir}/reconstructed_IoUs.npy", save_IoU_arr_ours)
np.save(f"./Output/{output_dir}/original_dices.npy", save_dice_arr_orig)
np.save(f"./Output/{output_dir}/reconstructed_dices.npy", save_dice_arr_ours)

np.save(f"./Output/ScalePoints{depth_model}.npy",scale_points_arr)
np.save(f"./Output/NormScalePoints{depth_model}.npy",norm_scale_points_arr)
np.save(f"./Output/MinMaxDepths{depth_model}.npy",min_max_depths_arr)

print(f"Average original IoU value: {np.average(save_IoU_arr_orig)}")
print(f"Average reconstructed IoU value: {np.average(save_IoU_arr_ours)}")
print(f"Average original Dices value: {np.average(save_dice_arr_orig)}")
print(f"Average reconstructed Dices value: {np.average(save_dice_arr_ours)}")

