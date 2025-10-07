import Depth_Map
import glob

images = glob.glob("ImageExtractorOutput/*.PNG")
total_images = len(images)
current_idx = 0

print(f"Generating: {len(images)} images")

for image in images:
    Depth_Map.generate_depth_map_as_rgb(image)
    current_idx += 1
    print(f"Completed depth estimation for: {image}, {current_idx}/{total_images}")

print("Completed Generating Depth Images")