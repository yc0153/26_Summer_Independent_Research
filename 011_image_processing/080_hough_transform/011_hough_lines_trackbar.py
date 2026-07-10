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
edges = cv2.Canny(image=blur, threshold1=50, threshold2=150)


"""===========================================================================================
Hough Line Trackbar
==========================================================================================="""
name_window = "hough_lines_trackbar"

cv2.namedWindow(name_window)
cv2.createTrackbar("vote", name_window, 70, 200, lambda value: None)
cv2.createTrackbar("min_length", name_window, 80, 300, lambda value: None)
cv2.createTrackbar("max_gap", name_window, 10, 80, lambda value: None)

while True:
    vote = max(1, cv2.getTrackbarPos("vote", name_window))
    min_length = max(1, cv2.getTrackbarPos("min_length", name_window))
    max_gap = cv2.getTrackbarPos("max_gap", name_window)

    result = img.copy()
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180,
                            threshold=vote, minLineLength=min_length,
                            maxLineGap=max_gap)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img=result, pt1=(x1, y1), pt2=(x2, y2),
                     color=(0, 160, 0), thickness=2)

    line_count = 0 if lines is None else len(lines)
    cv2.putText(img=result,
                text=f"vote={vote}, min_length={min_length}, "
                     f"gap={max_gap}, lines={line_count}",
                org=(10, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.65, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA)

    cv2.imshow(name_window, result)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output, "04_hough_lines_trackbar.png")
        cv2.imwrite(filename=path_save, img=result)
        print("Saved:", path_save)

cv2.destroyAllWindows()
