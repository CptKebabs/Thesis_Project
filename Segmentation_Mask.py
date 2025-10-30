#segmentation mask stuff here

import numpy
import cv2
from ultralytics import YOLO
 
def infer_and_create_mask(image_path, model_path, target_class):
    model = YOLO(model_path)

    image = cv2.imread(image_path)

    results = model.predict(image_path) #just model("bus.jpg works too") change to: model.predict(image_path,device="cpu") if the depthAnything problem keeps happening
                                        # in the future i can do batches to do a whole directory in one go with batch=2|4|8|16|32 etc
    class_names = model.names#dictionary

    final_mask = numpy.zeros((image.shape[0],image.shape[1]),dtype=bool)

    for result in results:#for each image
        
        #result.masks.data each detected object
        if result.masks is not None:
            for i, mask_data in enumerate(result.masks.data):
                # Convert mask to a NumPy array and resize to original image dimensions
                    class_indices = result.boxes.cls.cpu().tolist()
                    class_index = class_indices[i]
                    class_name = class_names[class_index]
                    #print(f"mask number: {i} is type: {class_name}")

                    mask = mask_data.cpu().numpy().astype(numpy.uint8)# move the mask_data tensor to cpu memory (.cpu()) then convert it to a ndarray (.numpy()) then to a 8 bit int 
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))# resize the mask to be same as image

                    #represent the mask only
                    if(class_name == target_class):#can do this if we run into issues with 
                        mask_rgb = numpy.zeros_like(image,dtype=numpy.uint8)
                        mask_rgb[mask == 1] = (255,255,255)
                        final_mask = numpy.logical_or(final_mask, mask)
                        display_mask(mask,image)

    display_mask(final_mask,image)
    return final_mask
                    

def testing_mask(height, width):
    print(width)
    print(height)
    mask = numpy.zeros((width, height), dtype=bool)
    mask[:int(height):][:int(width):] = True
    return mask


def display_mask(mask,image):
    rgb = numpy.zeros_like(image,dtype=numpy.uint8)
    rgb[mask == 1] = (255,255,255) 
    cv2.imshow("test",rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def reprojection_to_mask(reprojected_image):
    return numpy.any(reprojected_image != [0, 0, 0], axis=-1)#set mask pixel equivalent to True

def add_masks(mask1, mask2):
    return numpy.logical_or(mask1, mask2)#add them together

def calculate_IoU(mask1,mask2):# intersection (a AND b)/Union (a OR b)
    intersection = numpy.logical_and(mask1, mask2).sum()
    union = numpy.logical_or(mask1, mask2).sum()
    if union > 0:# case where neither detect anything
        return intersection / union
    return 1.0 #because both agree on there being nothing

#more stuff if needed https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
def fill_gaps_in_mask(mask,kernel_size,iter):#technically this is less correct than using a mesh produced from the point cloud but because our method can only ever produce a 2d cross section of the seaweed the 3d information produced by the mesh is mostly wasted anyway
    kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)#3x3 kernel for now

    new_mask = (mask.astype(numpy.uint8)) * 255#opencv wants int inputs so 255 being True 0 being false

    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel, iterations=iter)#Closing is Dilation followed by Erosion
    return new_mask > 0#back to boolean

def yolo_to_numpy(file_path, img_dimentions):#read a ground truth file return the numpy boolean mask 
    img_height = img_dimentions[0]
    img_width = img_dimentions[1]

    mask = numpy.zeros((img_height, img_width),  dtype=numpy.uint8)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:#for each mask
        parts = line.strip().split()#get rid of lead/trail whitespace then split by spaces
        if len(parts) < 3:#mask can be a single point
            print(f"Error could not read enough vertices to create polygon at: {file_path}")
            continue  # Not enough data for a polygon
        
        class_id = int(parts[0])#dont need this for my case
        coords = list(map(float, parts[1:]))#convert to float and add to list
        
        
        polygon = []
        for i in range(0, len(coords), 2):#for each x1 y1
            x = int(coords[i] * img_width)#unnormalise/convert to pixel values
            y = int(coords[i+1] * img_height)
            polygon.append([x, y])#add the point to the list
        
        polygon = numpy.array([polygon], dtype=numpy.int32)#convert to numpy
        
        #fill this polygons shape onto the mask array
        cv2.fillPoly(mask, polygon, 1)#fillpoly needs to work with numbers not booleans apparently

    mask = mask.astype(bool)#convert back to boolean mask
    return mask

#TODO:
#mAP logic - We are going to get the IoU values for each top mask generated, Then we get the Precision and Recall from the (TP FP FN for different IoU thresholds)
#          - then we can derive the precision and recall, generate the curve and find the area
#convert yolo format mask to numpy binary array

