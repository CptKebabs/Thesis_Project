import Depth_Map
import glob

images = glob.glob("ImageExtractorOutput/Test Pairs/*167.PNG")
total_images = len(images)
current_idx = 0

print(f"Generating: {len(images)} images")

for image in images:
    depth = Depth_Map.generate_relative_depth_map_depthanythingV2(image, normalise=True)
    Depth_Map.depth_to_rgb(depth,image)
    current_idx += 1
    print(f"Completed depth estimation for: {image}, {current_idx}/{total_images}")

print("Completed Generating Depth Images")