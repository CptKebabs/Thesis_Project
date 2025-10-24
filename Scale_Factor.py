#https://matplotlib.org/stable/users/explain/figure/event_handling.html

import matplotlib.pyplot as plt
import cv2
import glob
import os

GLOB_PATH = "./*.png" #will read from and save to same directory

def on_click(event):#event handler for clicking the display
    if event.button == 1:
        # colour_val = image[int(event.ydata),int(event.xdata)]
        # r = int(colour_val[0])
        # g = int(colour_val[1])
        # b = int(colour_val[2])
        #colour_val = (r,g,b)
        depth_val = 0 #cmap_list.index((r,g,b))not gonna use this its only for normalised 0-255
        # print(f"{colour_val}")
        # print(f"Mouse clicked at image coordinates: x={int(event.ydata)}, y={int(event.xdata)}")#actual window coordinates is just event.x or y
        #print(f"Depth Value at pixel: {depth_val}")
        # scale_factor = get_scale_factor(depth_val, REFERENCE_OBJECT_DIST)
        # print(f"Scale factor is: {scale_factor}")

        #output_file = os.path.basename(curr_image)
        output_file = f"{os.path.splitext(curr_image)[0]}.scale"

        global click_count # to modify 
        if click_count == 0:
            with open(f"{output_file}","w") as file:#create the file or overwrite it
                file.write(f"({int(event.xdata)},{int(event.ydata)}) = {depth_val}\n")
                print(f"Wrote: {int(event.xdata)},{int(event.ydata)}  = {depth_val}")
            click_count = 1
        elif click_count == 1:#append to the file for the second point
                with open(f"{output_file}","a") as file:
                    file.write(f"({int(event.xdata)},{int(event.ydata)}) = {depth_val}")
                    print(f"Wrote: {int(event.xdata)},{int(event.ydata)} = {depth_val}")
                click_count = 0
                plt.close()


# #cmap = plt.get_cmap('Spectral_r')#This is the colourmap we are saving the depth images with
# cmap_for_list = plt.get_cmap('Spectral')#we will use this one for getting our list so that low values are closer

# #first build a list so we can get depth from the colour mapped image (as far as i know there is no premade reversal function)
# cmap_list = []#build colour map values here for lookup
# for i in range(256):
#     r = int(cmap_for_list(i)[0]*255)
#     g = int(cmap_for_list(i)[1]*255)
#     b = int(cmap_for_list(i)[2]*255)
#     cmap_list.append((r,g,b))
#     print(f"depth: ({i}) = {(r,g,b)}")

#do glob stuff here to complete for every file in folder (top then bottom for each file and after the bottom is complete we write to a file the scale factors for both)
depth_images = glob.glob(GLOB_PATH)
curr_image = ""     #global variable
click_count = 0     #global variable

if len(depth_images) > 0:
    curr_image = depth_images[0]
else:
    print("No depth images found")
    exit(0)

for depth_image in depth_images:#open each image and view with our event listener logic
    curr_image = depth_image
    click_count = 0

    print(f"loaded: {depth_image}")
    fig, ax = plt.subplots()
    ax.axis("off")
    image = cv2.imread(depth_image)[:, :, ::-1]#bgr to rgb
    im = ax.imshow(image)

    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)#add event listener

    plt.show()

    fig.canvas.mpl_disconnect(cid_click)#remove event listener