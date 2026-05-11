import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "050_morphology")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "mask_noisy.png")


"""===========================================================================================
Create Binary Mask Input
==========================================================================================="""
if not os.path.exists(path_input):
    mask = np.zeros(shape=(320, 520), dtype=np.uint8)
    cv2.rectangle(img=mask, pt1=(80, 80), pt2=(240, 230),
                  color=255, thickness=-1)
    cv2.circle(img=mask, center=(365, 155), radius=80,
               color=255, thickness=-1)
    cv2.circle(img=mask, center=(365, 155), radius=28,
               color=0, thickness=-1)

    rng = np.random.default_rng(seed=5)
    for x, y in rng.integers(low=[0, 0], high=[520, 320],
                             size=(120, 2)):
        mask[y, x] = 255

    cv2.imwrite(filename=path_input, img=mask)

mask = cv2.imread(filename=path_input, flags=cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Morphological Gradient
==========================================================================================="""
# Morphological gradient is dilation minus erosion.
kernel = np.ones(shape=(5, 5), dtype=np.uint8)
gradient = cv2.morphologyEx(src=mask, op=cv2.MORPH_GRADIENT,
                            kernel=kernel, iterations=1)

cv2.imwrite(filename=os.path.join(dir_output, "01_mask_noisy.png"),
            img=mask)
cv2.imwrite(filename=os.path.join(dir_output, "06_morph_gradient.png"),
            img=gradient)

print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
