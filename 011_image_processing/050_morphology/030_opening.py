import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "050_morphology")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "mask_opening.png")


"""===========================================================================================
Create Binary Mask Input
==========================================================================================="""
if not os.path.exists(path_input):
    # Opening is easiest to see when a thin white bridge gets removed.
    mask = np.zeros(shape=(320, 520), dtype=np.uint8)
    cv2.circle(img=mask, center=(185, 160), radius=65,
               color=255, thickness=-1)
    cv2.circle(img=mask, center=(335, 160), radius=65,
               color=255, thickness=-1)
    cv2.rectangle(img=mask, pt1=(185, 158), pt2=(335, 161),
                  color=255, thickness=-1)

    rng = np.random.default_rng(seed=5)
    for cx, cy in rng.integers(low=[0, 0], high=[520, 320],
                               size=(60, 2)):
        cv2.circle(img=mask, center=(int(cx), int(cy)), radius=2,
                   color=255, thickness=-1)

    cv2.imwrite(filename=path_input, img=mask)

mask = cv2.imread(filename=path_input, flags=cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Opening
==========================================================================================="""
# Opening is erosion followed by dilation.
kernel = np.ones(shape=(5, 5), dtype=np.uint8)
opened = cv2.morphologyEx(src=mask, op=cv2.MORPH_OPEN,
                          kernel=kernel, iterations=1)

cv2.imwrite(filename=os.path.join(dir_output, "04a_opening_input.png"),
            img=mask)
cv2.imwrite(filename=os.path.join(dir_output, "04b_opened.png"), img=opened)

print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
