import sys
import matplotlib
import torch
import os
import cv2
import numpy
import re
from PIL import Image

INPUT_SIZE = 750 #for depthanything
REF_OBJ_SIZE = .2 #metres
REF_OBJ_DIST = .362 #metres


def generate_metric_depth_map_depthanythingV2(raw_img_cv, outdoor = True):

    sys.path.append(os.path.abspath('../Depth-Anything-V2/metric_depth'))
    from depth_anything_v2.dpt import DepthAnythingV2

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl' # or 'vits', 'vitb'
    if outdoor:
        dataset = 'vkitti' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 80 # 20 for indoor model, 80 for outdoor model
    else:
        dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
        max_depth = 20 # 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(f'DepthModel/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location=device))
    model.to(device)
    model.eval()



    #raw_img = cv2.imread(image_path)
    depth = model.infer_image(raw_img_cv,INPUT_SIZE) # HxW depth map in meters in numpy

    print(f"Minimum depth value: {depth.min()}")
    print(f"Maximum depth value: {depth.max()}")

    return depth

def generate_relative_depth_map_depthanythingV2(raw_img_cv, normalise):

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

    #raw_img = cv2.imread(image_path)

    depth = model.infer_image(raw_img_cv,INPUT_SIZE) #can vary the input size as a second parameter here default is probably 518# HxW raw depth map in numpy
    print(f"Minimum depth value: {depth.min()}")
    print(f"Maximum depth value: {depth.max()}")

    if normalise:
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0 #distribiute values between 0 and 255
        depth = depth.astype(numpy.uint8)#change the array to type uint8 (0-255)
        #if reference object values are too close to 0 we may need to allow the 0-min and max-255 values


    return depth

def generate_metric_depth_map_depthpro(raw_img_PIL, focal_length):
    import depth_pro

    #load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    #load and preprocess an image.
    #image, _, f_px = depth_pro.load_rgb(image_path)#also a predicted focal length but dont see the point in using it
    image = numpy.array(raw_img_PIL)
    image = transform(image)

    #run inference.
    prediction = model.infer(image, f_px=torch.tensor(focal_length))
    depth = prediction["depth"]
    focallength_px = prediction["focallength_px"]#focal length

    print(depth.shape)
    print(f"Minimum depth value: {depth.min()}")
    print(f"Maximum depth value: {depth.max()}")
    #print(focallength_px.item())#extracts the number from the tensor

    return depth.cpu().numpy()


def generate_relative_depthmap_UDepth(raw_img_PIL, mask):

    sys.path.append(os.path.abspath('../UDepth'))
    from model.udepth import UDepth
    #from utils.data import RGB_to_RMI
    from utils.utils import output_result

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = UDepth.build(n_bins=80, min_val=0.001, max_val=1, norm="linear")
    net.load_state_dict(torch.load("./DepthModel/model_RGB.pth"))
    net.to(device)
    net.eval()

    img = raw_img_PIL.resize((640,480),Image.BILINEAR)
    img = numpy.array(img).astype(numpy.float32) / 255.0
    
    
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)#change dimensions and then wrap in extra brackets and send to device

    with torch.no_grad():#we dont want to modify model, faster and more efficient
        _, depth = net(img)

    resized_mask = cv2.resize(mask.astype(numpy.uint8), (320,240), interpolation = cv2.INTER_NEAREST)
    resized_mask = resized_mask.astype(bool)
    depth = output_result(depth, resized_mask)# this does the tensor squeeze cpu numpy stuff, its using the mask to make the depths more consistent in that region
    return depth













def depth_to_rgb(depth,image_path):
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


def read_scale_file(path):
    ret_points = []
    ret_values = []

    regex = r"\((\d+),(\d+)\)\s*=\s*(\d+)"

    with open(path, "r") as file:
        for line in file:
            match = re.search(regex,line)
            if match:
                x = match.group(1)
                y = match.group(2)
                depth_val = match.group(3)
                ret_points.append((x,y))
                ret_values.append(depth_val)

    return ret_points, ret_values


def get_simple_scale_factor(depth_map,x,y):#simple scale factor
    return REF_OBJ_DIST / depth_map[y][x]#real distance / the predicted distance

def get_scale_factor(depth_map, x1, y1, x2, y2, fx, fy, cx, cy): #scale factor via 3d distance
    point_1 = pixel_to_3d(x1,y1,depth_map,fx,fy,cx,cy)
    point_2 = pixel_to_3d(x2,y2,depth_map,fx,fy,cx,cy)
    apparent_length = numpy.linalg.norm(point_1 - point_2)#euclidean distance
    return REF_OBJ_SIZE / apparent_length, point_1[2], point_2[2] # the scale factor from the reference object coordinates from real size 

def pixel_to_3d(u, v, depth_map, fx, fy, cx, cy):#same logic as in pointcloud
    z = depth_map[v][u] #u,v or v,u (depth is in format y,x) but keep in mind that the .scale files are also in this format
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    print(f"depth value at point: {z}")
    return numpy.array([x, y, z])