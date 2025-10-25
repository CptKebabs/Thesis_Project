# Thesis_Project
3D Reconstruction Program
Guven Gemikonakli

## Requirements:
- Python Version: 3.9.12
- opencv-python Version: 4.11.0.86
- open3d Version: 0.19.0
- ultralytics Version: 8.3.202
- pillow Version: 11.3.0


## Setup Instructions
- pip install requirements.txt
- clone DepthAnythingV2 at https://github.com/DepthAnything/Depth-Anything-V2 into the same directory as this repo (not inside this repo)
- Download one of the or both the large metric depthanything models on the repo vikitti or hypersim (make sure code is changed appropriately)
- Download one of the yolo segmentation pretrained models and place inside the Segmentation directory (not needed right now)

## Usage Instructions
Main Execution takes place inside Run.py
### Image Pairs
- if you are extracting from videos make sure your two videos for extraction (which are syncronised to be at the same time) are inside the Videos directory
- change the paths of the videos inside the dual_perspective_extractor.py to match your videos and run the script (s to save image, q to quit)

### Scale Factor
- Use scale factor to generate reference object points for point cloud scaling



## Current todos
- B3MD?
- Fix Gaps in reprojections