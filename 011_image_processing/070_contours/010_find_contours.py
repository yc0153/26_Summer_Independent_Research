import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "070_contours")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "shapes.png")


"""===========================================================================================
Create Shape Input
==========================================================================================="""
if not os.path.exists(path_input):
    img = np.full(shape=(360, 560, 3), fill_value=255, dtype=np.uint8)
    cv2.circle(img=img, center=(105, 115), radius=55,
               color=(40, 40, 40), thickness=-1)
    cv2.rectangle(img=img, pt1=(230, 65), pt2=(365, 180),
                  color=(40, 40, 40), thickness=-1)

    triangle = np.array(object=[[455, 55], [390, 185], [520, 185]])
    cv2.drawContours(image=img, contours=[triangle], contourIdx=-1,
                     color=(40, 40, 40), thickness=-1)

    cv2.circle(img=img, center=(180, 275), radius=45,
               color=(40, 40, 40), thickness=-1)
    cv2.rectangle(img=img, pt1=(330, 235), pt2=(480, 320),
                  color=(40, 40, 40), thickness=-1)
    cv2.imwrite(filename=path_input, img=img)

img = cv2.imread(filename=path_input)
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Find Contours
==========================================================================================="""
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(src=gray, thresh=120, maxval=255,
                          type=cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(
    image=binary, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()
cv2.drawContours(image=result, contours=contours, contourIdx=-1,
                 color=(0, 0, 255), thickness=2)

cv2.imwrite(filename=os.path.join(dir_output, "01_input_shapes.png"), img=img)
cv2.imwrite(filename=os.path.join(dir_output, "02_binary.png"), img=binary)
cv2.imwrite(filename=os.path.join(dir_output, "03_contours.png"), img=result)

print("Contours:", len(contours))
print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
