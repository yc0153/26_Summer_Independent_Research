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
Sobel Trackbar
==========================================================================================="""
name_window = "sobel_trackbar"

# ksize slider: 0,1,2,3 -> 1,3,5,7
# direction: 0 = x+y, 1 = x only, 2 = y only
cv2.namedWindow(name_window)
cv2.createTrackbar("ksize", name_window, 1, 3, lambda value: None)
cv2.createTrackbar("direction", name_window, 0, 2, lambda value: None)

while True:
    ksize = cv2.getTrackbarPos("ksize", name_window) * 2 + 1
    direction = cv2.getTrackbarPos("direction", name_window)

    if direction == 1:
        sobel = cv2.Sobel(src=blur, ddepth=cv2.CV_64F,
                          dx=1, dy=0, ksize=ksize)
        result = cv2.convertScaleAbs(src=sobel)
        label = f"sobel x, ksize={ksize}"

    elif direction == 2:
        sobel = cv2.Sobel(src=blur, ddepth=cv2.CV_64F,
                          dx=0, dy=1, ksize=ksize)
        result = cv2.convertScaleAbs(src=sobel)
        label = f"sobel y, ksize={ksize}"

    else:
        sobel_x = cv2.Sobel(src=blur, ddepth=cv2.CV_64F,
                            dx=1, dy=0, ksize=ksize)
        sobel_y = cv2.Sobel(src=blur, ddepth=cv2.CV_64F,
                            dx=0, dy=1, ksize=ksize)
        result = cv2.convertScaleAbs(src=np.abs(sobel_x) + np.abs(sobel_y))
        label = f"sobel x+y, ksize={ksize}"

    preview = np.hstack(tup=[
        add_label(gray_image=gray, text="input"),
        add_label(gray_image=result, text=label)
    ])

    cv2.imshow(name_window, preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output, "04_sobel_trackbar.png")
        cv2.imwrite(filename=path_save, img=preview)
        print("Saved:", path_save)

cv2.destroyAllWindows()
