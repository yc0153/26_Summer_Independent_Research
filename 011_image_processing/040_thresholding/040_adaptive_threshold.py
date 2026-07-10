import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "040_thresholding")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "threshold_sample.png")


"""===========================================================================================
Create Input Image
==========================================================================================="""
if not os.path.exists(path_input):
    height, width = 320, 520
    gradient = np.linspace(start=220, stop=110, num=width, dtype=np.uint8)
    sample = np.tile(A=gradient, reps=(height, 1))

    cv2.rectangle(img=sample, pt1=(70, 70), pt2=(210, 220),
                  color=50, thickness=-1)
    cv2.circle(img=sample, center=(360, 150), radius=70,
               color=70, thickness=-1)
    cv2.putText(img=sample, text="CV", org=(235, 285),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2.2,
                color=45, thickness=5)
    cv2.imwrite(filename=path_input, img=sample)

gray = cv2.imread(filename=path_input, flags=cv2.IMREAD_GRAYSCALE)
if gray is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Adaptive Threshold
==========================================================================================="""
# Adaptive thresholding uses local brightness.
# Gaussian can leave only the border of a large object.
# Mean is easier to understand first because it keeps this sample filled.
adaptive = cv2.adaptiveThreshold(
    src=gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
    thresholdType=cv2.THRESH_BINARY_INV, blockSize=151, C=5)

cv2.imwrite(filename=os.path.join(dir_output, "01_gray.png"), img=gray)
cv2.imwrite(filename=os.path.join(dir_output, "05_adaptive_threshold.png"),
            img=adaptive)

print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
