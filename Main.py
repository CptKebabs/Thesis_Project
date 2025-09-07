# Main program, will utilise the other subsystems together
import subprocess
import os

# Determine the command based on the operating system
if os.name == 'nt':  # Windows
    command = 'dir'
else:  # Unix-like systems (Linux, macOS)
    command = 'ls'

try:
    result = subprocess.run(#this is blocking
        command,
        cwd='./',             # Current working directory
        capture_output=True,  # Capture standard output and error
        text=True,            # Return output as text (instead of bytes)
        check=True,           # Raise exception for non-zero exit codes
        shell=True            # Run the command through the shell
    )
    print("Command output:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Command failed with error: {e}")
    print(f"Error output: {e.stderr}")  # Print the standard error output

#Process will be as follows:
    #loop for all pairs of images (first get it working for one then repeat for all in input folder)(
        #preprocess with B3MD (top and bottom)

        #Generate point clouds with DepthAnythingV2 and scale them using a reference object in frame (top and bottom)

        #run segmentation algorithm YOLO(top and bottom)

        #merge point clouds with segmentation algorithm (top and bottom)

        #merge top and bottom coordinate frames (rotation and translation)

        #reproject into each perspective (top and bottom)

        #write point cloud and segmentations to output folder
    #)end loop

    #have something here to compare the original segmentations with the ones this program has created

#Assumptions:
    
    #needs a trained segmentation algorithm

    #will need a way to compare to segmentation algorithm alone

    #images need an object of known size to exist within it to be able to determine scale