import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "050_morphology")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "mask_closing.png")


"""===========================================================================================
Create Binary Mask Input
==========================================================================================="""
if not os.path.exists(path_input):
    # Closing is easiest to see with small black holes and narrow gaps.
    mask = np.zeros(shape=(320, 520), dtype=np.uint8)
    cv2.rectangle(img=mask, pt1=(80, 80), pt2=(240, 230),
                  color=255, thickness=-1)
    cv2.circle(img=mask, center=(365, 155), radius=80,
               color=255, thickness=-1)
    cv2.line(img=mask, pt1=(80, 150), pt2=(240, 150),
             color=0, thickness=5)
    cv2.line(img=mask, pt1=(365, 75), pt2=(365, 235),
             color=0, thickness=5)

    rng = np.random.default_rng(seed=7)
    for cx, cy in rng.integers(low=[95, 95], high=[430, 220],
                               size=(24, 2)):
        cv2.circle(img=mask, center=(int(cx), int(cy)), radius=4,
                   color=0, thickness=-1)

    cv2.imwrite(filename=path_input, img=mask)

mask = cv2.imread(filename=path_input, flags=cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Closing
==========================================================================================="""
# Closing is dilation followed by erosion.
kernel = np.ones(shape=(5, 5), dtype=np.uint8)
closed = cv2.morphologyEx(src=mask, op=cv2.MORPH_CLOSE,
                          kernel=kernel, iterations=1)

cv2.imwrite(filename=os.path.join(dir_output, "05a_closing_input.png"),
            img=mask)
cv2.imwrite(filename=os.path.join(dir_output, "05b_closed.png"), img=closed)

print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
