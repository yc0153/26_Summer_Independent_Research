import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing",
                        "090_corner_detection")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "corner_sample.png")


"""===========================================================================================
Create Corner Input
==========================================================================================="""
if not os.path.exists(path_input):
    img = np.full(shape=(480, 720, 3), fill_value=255, dtype=np.uint8)

    cv2.rectangle(img=img, pt1=(70, 70), pt2=(240, 220),
                  color=(40, 40, 40), thickness=-1)
    cv2.rectangle(img=img, pt1=(95, 285), pt2=(270, 420),
                  color=(40, 40, 40), thickness=7)

    l_shape = np.array(object=[[390, 70], [620, 70], [620, 140],
                               [465, 140], [465, 285], [390, 285]],
                       dtype=np.int32)
    cv2.drawContours(image=img, contours=[l_shape], contourIdx=-1,
                     color=(40, 40, 40), thickness=-1)

    cv2.rectangle(img=img, pt1=(520, 320), pt2=(640, 420),
                  color=(40, 40, 40), thickness=-1)

    cv2.imwrite(filename=path_input, img=img)

img = cv2.imread(filename=path_input)
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Harris Corner Detection
==========================================================================================="""
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
gray_float = np.float32(gray)

response = cv2.cornerHarris(src=gray_float, blockSize=2, ksize=3, k=0.04)
response = cv2.dilate(src=response, kernel=None)

response_norm = cv2.normalize(src=response, dst=None, alpha=0, beta=255,
                              norm_type=cv2.NORM_MINMAX)
response_norm = np.uint8(response_norm)

_, corner_mask = cv2.threshold(src=response_norm, thresh=125,
                               maxval=255, type=cv2.THRESH_BINARY)
_, _, _, centers = cv2.connectedComponentsWithStats(image=corner_mask)

result = img.copy()
for center in centers[1:]:
    x, y = np.round(center).astype(np.int32)
    cv2.circle(img=result, center=(int(x), int(y)), radius=5,
               color=(0, 0, 255), thickness=-1)

cv2.imwrite(filename=os.path.join(dir_output, "01_input_corner_sample.png"),
            img=img)
cv2.imwrite(filename=os.path.join(dir_output, "02_harris_response.png"),
            img=response_norm)
cv2.imwrite(filename=os.path.join(dir_output, "03_harris_corners.png"),
            img=result)

print("Harris corners:", max(0, len(centers) - 1))
print("Saved input image to:", path_input)
print("Saved output images to:", dir_output)
