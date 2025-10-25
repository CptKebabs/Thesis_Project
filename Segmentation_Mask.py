#segmentation mask stuff here

import numpy
import cv2
from ultralytics import YOLO

def infer_and_create_mask(image_path):
    model = YOLO("Segmentation/yolo11m-seg.pt")#this is a pretrained model on the COCO dataset

    #image_path = "./Thesis_Project/bus.jpg"

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
                    if(class_name == "person"):#can do this if we run into issues with 
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

def calculate_IoU(mask1,mask2):# intersection (a OR b)/Union
    intersection = numpy.logical_and(mask1, mask2).sum()
    union = numpy.logical_or(mask1, mask2).sum()

    return intersection / union

