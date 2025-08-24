# Main program, will utilise the other subsystems together

#Process will be as follows:

    #preprocess first (top and bottom)

    #Generate point clouds (top and bottom)

    #run segmentation algorithm (top and bottom)

    #merge point clouds with segmentation algorithm (top and bottom)

    #merge top and bottom coordinate frames

    #reproject into each perspective (top and bottom)

#Assumptions:
    
    #needs a trained segmentation algorithm

    #will need a way to compare to segmentation algorithm alone