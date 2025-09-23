#Based on:
#https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
#https://github.com/niconielsen32/cameracalibration

import glob
import numpy
import cv2

grid_size = (9,6)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)#if epsilon is less than a threshold or if we do 30 iterations end the process

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = numpy.zeros((grid_size[0]*grid_size[1],3), numpy.float32)#6: width inner square corners 
objp[:,:2] = numpy.mgrid[0:grid_size[1],0:grid_size[0]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

Calibration_Images = glob.glob("./CalibrationInput/*.png")#Every Image in calibration input

for Calibration_Image in Calibration_Images:

    print(f"processing:{Calibration_Image}")
    
    img = cv2.imread(Calibration_Image)
    grey_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(grey_image, (grid_size[1],grid_size[0]))
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        print("Located Grid")
        objpoints.append(objp)
 
        corners2 = cv2.cornerSubPix(grey_image,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (grid_size[1],grid_size[0]), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to locate grid")
    #Do calibration stuff

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, grey_image.shape[::-1], None, None)

print(mtx)

#1080p 16:9 W 
#earlier
#[[887.96476705   0.         960.38019362]
# [  0.         887.51688746 532.69955894]
# [  0.           0.           1.        ]]

#later
#[[888.29855742   0.         960.53009391]
# [  0.         887.67732304 532.68979747]
# [  0.           0.           1.        ]]