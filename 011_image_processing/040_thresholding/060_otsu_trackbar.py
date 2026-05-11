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
Otsu Trackbar
==========================================================================================="""
name_window = "otsu_trackbar"

# Otsu chooses the threshold automatically, so we change blur instead.
cv2.namedWindow(name_window)
cv2.createTrackbar("blur", name_window, 0, 10, lambda value: None)
cv2.createTrackbar("inverse", name_window, 0, 1, lambda value: None)

while True:
    blur = cv2.getTrackbarPos("blur", name_window)
    inverse = cv2.getTrackbarPos("inverse", name_window)

    ksize = blur * 2 + 1
    if ksize == 1:
        src = gray
    else:
        src = cv2.GaussianBlur(src=gray, ksize=(ksize, ksize), sigmaX=0)

    mode = cv2.THRESH_BINARY_INV if inverse == 1 else cv2.THRESH_BINARY
    threshold, result = cv2.threshold(src=src, thresh=0, maxval=255,
                                      type=mode + cv2.THRESH_OTSU)

    preview = cv2.cvtColor(src=result, code=cv2.COLOR_GRAY2BGR)
    cv2.putText(img=preview, text=f"otsu={threshold:.0f}, blur={ksize}",
                org=(10, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA)

    cv2.imshow(name_window, preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output, "06_otsu_trackbar.png")
        cv2.imwrite(filename=path_save, img=result)
        print("Saved:", path_save)

cv2.destroyAllWindows()
