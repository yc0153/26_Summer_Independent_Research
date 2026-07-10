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
Hough Line Transform
==========================================================================================="""
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
edges = cv2.Canny(image=blur, threshold1=50, threshold2=150)

result = img.copy()
lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180,
                        threshold=70, minLineLength=80, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img=result, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 160, 0), thickness=3)

line_count = 0 if lines is None else len(lines)

cv2.imwrite(filename=os.path.join(dir_output, "01_input_geometry.png"),
            img=img)
cv2.imwrite(filename=os.path.join(dir_output, "02_edges.png"), img=edges)
cv2.imwrite(filename=os.path.join(dir_output, "03_hough_lines.png"),
            img=result)

print("Lines:", line_count)
print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
