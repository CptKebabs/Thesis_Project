import sys
import matplotlib
import torch
import os
import cv2
import numpy

INPUT_SIZE = 500


def generate_depth_map(image_path):

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

    img = cv2.imread(image_path)

    depth = model.infer_image(img,INPUT_SIZE) #can vary the input size as a second parameter here default is probably 518# HxW raw depth map in numpy

    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 #distribiute values between 0 and 255
    depth = depth.astype(numpy.uint8)#change the array to type uint8 (0-255)
    
    return depth

def generate_depth_map_as_rgb(image_path):
    depth = generate_depth_map(image_path)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')#spectral reversed 
    #print(depth)
    depthImage = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(numpy.uint8)
            #     cmap(depth) - convert the depthMap to a coloured image by passing it through a colour map     
            #                [:, :, :3] - extract rgb only from colourmap output,
                                # * 255 because output is 0-1 range
                                        #[:, :, ::-1] reverse the order because its in BGR
                                                    #.astype(numpy.uint8) convert to int 0-255 instead of floats
    #print(type(depthImage))

    cv2.imwrite(os.path.join("./DepthImages", os.path.splitext(os.path.basename(f"{image_path}"))[0] + f"_{INPUT_SIZE}.png"), depthImage)#save this new depth image to depthImages


if __name__ == "__main__":
    print("running as main")