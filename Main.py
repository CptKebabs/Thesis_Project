# Main program, will utilise the other subsystems together

#Process will be as follows:
    #loop for all pairs of images(
        #preprocess with B3MD (top and bottom)

        #Generate point clouds with DepthAnythingV2 (top and bottom)

        #run segmentation algorithm YOLO(top and bottom)

        #merge point clouds with segmentation algorithm (top and bottom)

        #merge top and bottom coordinate frames (rotation and translation)

        #reproject into each perspective (top and bottom)
    #)end loop

#Assumptions:
    
    #needs a trained segmentation algorithm

    #will need a way to compare to segmentation algorithm alone

    # images need an object of known size to exist within it to be able to determine scale