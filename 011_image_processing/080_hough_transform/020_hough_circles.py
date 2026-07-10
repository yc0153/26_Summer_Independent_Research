import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing",
                        "080_hough_transform")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "geometry.png")


"""===========================================================================================
Create Geometric Input
==========================================================================================="""
if not os.path.exists(path_input):
    img = np.full(shape=(480, 720, 3), fill_value=255, dtype=np.uint8)

    cv2.line(img=img, pt1=(70, 90), pt2=(640, 90),
             color=(40, 40, 40), thickness=4)
    cv2.line(img=img, pt1=(95, 390), pt2=(630, 140),
             color=(40, 40, 40), thickness=4)
    cv2.line(img=img, pt1=(95, 140), pt2=(630, 390),
             color=(40, 40, 40), thickness=4)

    cv2.rectangle(img=img, pt1=(80, 235), pt2=(235, 390),
                  color=(40, 40, 40), thickness=4)
    cv2.rectangle(img=img, pt1=(420, 245), pt2=(600, 395),
                  color=(40, 40, 40), thickness=4)

    cv2.circle(img=img, center=(335, 165), radius=55,
               color=(40, 40, 40), thickness=4)
    cv2.circle(img=img, center=(335, 335), radius=65,
               color=(40, 40, 40), thickness=4)
    cv2.imwrite(filename=path_input, img=img)

img = cv2.imread(filename=path_input)
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Hough Circle Transform
==========================================================================================="""
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

result = img.copy()
circles = cv2.HoughCircles(
    image=blur, method=cv2.HOUGH_GRADIENT, dp=1.2, minDist=90,
    param1=100, param2=28, minRadius=35, maxRadius=80)

if circles is not None:
    circles = np.round(circles[0]).astype(np.int32)
    for x, y, radius in circles:
        cv2.circle(img=result, center=(x, y), radius=radius,
                   color=(0, 0, 255), thickness=3)
        cv2.circle(img=result, center=(x, y), radius=4,
                   color=(255, 0, 0), thickness=-1)

circle_count = 0 if circles is None else len(circles)

cv2.imwrite(filename=os.path.join(dir_output, "01_input_geometry.png"),
            img=img)
cv2.imwrite(filename=os.path.join(dir_output, "05_hough_circles.png"),
            img=result)

print("Circles:", circle_count)
print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
