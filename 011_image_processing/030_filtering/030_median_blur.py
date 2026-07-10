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
Create Noisy Input
==========================================================================================="""
# Add random noise so the blur effect is easy to compare.
rng = np.random.default_rng(seed=10)
noise = rng.normal(loc=0, scale=15, size=img.shape)
noisy = np.clip(a=img.astype(np.float32) + noise,
                a_min=0, a_max=255).astype(np.uint8)

cv2.imwrite(filename=os.path.join(dir_output, "01_noisy.png"), img=noisy)


"""===========================================================================================
Median Blur
==========================================================================================="""
# Median blur is useful for dot-like noise.
median = cv2.medianBlur(src=noisy, ksize=9)

cv2.imwrite(filename=os.path.join(dir_output, "04_median_blur.png"),
            img=median)

print("Saved output images to:", dir_output)
