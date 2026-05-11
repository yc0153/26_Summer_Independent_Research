import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "030_filtering")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(filename=os.path.join(dir_input, "flower.png"))
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Sharpening
==========================================================================================="""
# A sharpening kernel emphasizes the center pixel and subtracts nearby pixels.
kernel_sharpen = np.float32([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

sharpened = cv2.filter2D(src=img, ddepth=-1, kernel=kernel_sharpen)

cv2.imwrite(filename=os.path.join(dir_output, "05_sharpened.png"),
            img=sharpened)

print("Saved output images to:", dir_output)
