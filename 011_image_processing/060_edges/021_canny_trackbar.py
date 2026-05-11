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

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)


"""===========================================================================================
Helper Function
==========================================================================================="""
def add_label(gray_image, text):
    result = cv2.cvtColor(src=gray_image, code=cv2.COLOR_GRAY2BGR)
    cv2.putText(img=result, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return result


"""===========================================================================================
Canny Trackbar
==========================================================================================="""
name_window = "canny_trackbar"

cv2.namedWindow(name_window)
cv2.createTrackbar("low", name_window, 50, 255, lambda value: None)
cv2.createTrackbar("high", name_window, 150, 255, lambda value: None)

while True:
    low = cv2.getTrackbarPos("low", name_window)
    high = cv2.getTrackbarPos("high", name_window)
    high = max(high, low + 1)

    result = cv2.Canny(image=blur, threshold1=low, threshold2=high)
    preview = np.hstack(tup=[
        add_label(gray_image=gray, text="input"),
        add_label(gray_image=result, text=f"canny {low}/{high}")
    ])

    cv2.imshow(name_window, preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output, "05_canny_trackbar.png")
        cv2.imwrite(filename=path_save, img=preview)
        print("Saved:", path_save)

cv2.destroyAllWindows()
