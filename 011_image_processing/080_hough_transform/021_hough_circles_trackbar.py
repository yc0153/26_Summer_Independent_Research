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

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)


"""===========================================================================================
Hough Circle Trackbar
==========================================================================================="""
name_window = "hough_circles_trackbar"

cv2.namedWindow(name_window)
cv2.createTrackbar("vote", name_window, 28, 100, lambda value: None)
cv2.createTrackbar("min_radius", name_window, 35, 100, lambda value: None)
cv2.createTrackbar("max_radius", name_window, 80, 140, lambda value: None)

while True:
    vote = max(1, cv2.getTrackbarPos("vote", name_window))
    min_radius = max(1, cv2.getTrackbarPos("min_radius", name_window))
    max_radius = max(min_radius + 1,
                     cv2.getTrackbarPos("max_radius", name_window))

    result = img.copy()
    circles = cv2.HoughCircles(
        image=blur, method=cv2.HOUGH_GRADIENT, dp=1.2, minDist=90,
        param1=100, param2=vote,
        minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = np.round(circles[0]).astype(np.int32)
        for x, y, radius in circles:
            cv2.circle(img=result, center=(x, y), radius=radius,
                       color=(0, 0, 255), thickness=2)
            cv2.circle(img=result, center=(x, y), radius=4,
                       color=(255, 0, 0), thickness=-1)

    circle_count = 0 if circles is None else len(circles)
    cv2.putText(img=result,
                text=f"vote={vote}, radius={min_radius}-{max_radius}, "
                     f"circles={circle_count}",
                org=(10, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.65, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA)

    cv2.imshow(name_window, result)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output,
                                 "06_hough_circles_trackbar.png")
        cv2.imwrite(filename=path_save, img=result)
        print("Saved:", path_save)

cv2.destroyAllWindows()
