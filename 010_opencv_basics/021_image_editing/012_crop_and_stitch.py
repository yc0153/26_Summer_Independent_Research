import os
import cv2

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "010_opencv_basics",
                        "021_image_editing")
img_path = os.path.join(dir_base, "input", "sample.png")
dir_output = os.path.join(dir_base, "output", "crop_and_stitch")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(img_path)
if img is None:
    print("Error: Could not open image")
    exit()

# Check the heigh and width of image
height, width = img.shape[:2]
y_points = [0, height // 2, height]
x_points = [0, width // 2, width]

"""===========================================================================================
Cropping
==========================================================================================="""
# Crop the image into a 2x2 grid and save each piece.
crops = []
for y in range(2):
    row = []
    for x in range(2):
        crop = img[y_points[y]:y_points[y + 1],
                   x_points[x]:x_points[x + 1]]
        row.append(crop)
        cv2.imwrite(os.path.join(dir_output, f"crop_{y}_{x}.png"), crop)
    crops.append(row)

"""===========================================================================================
Stitching
==========================================================================================="""
# Stitch the cropped pieces back together horizontally and vertically.
top = cv2.hconcat(crops[0])
bottom = cv2.hconcat(crops[1])
stitched = cv2.vconcat([top, bottom])

# Save the final stitched image.
cv2.imwrite(os.path.join(dir_output, "stitched.png"), stitched)
