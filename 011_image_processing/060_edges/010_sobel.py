import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "060_edges")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(filename=os.path.join(dir_input, "flower.png"))
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Sobel Edge Detection
==========================================================================================="""
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

# Sobel detects brightness changes in the x and y directions.
sobel_x = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_y = cv2.Sobel(src=blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
sobel = cv2.convertScaleAbs(src=np.abs(sobel_x) + np.abs(sobel_y))

cv2.imwrite(filename=os.path.join(dir_output, "01_gray.png"), img=gray)
cv2.imwrite(filename=os.path.join(dir_output, "02_sobel.png"), img=sobel)

print("Saved output images to:", dir_output)
