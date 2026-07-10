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

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
gray_float = np.float32(gray)


"""===========================================================================================
Harris Corner Trackbar
==========================================================================================="""
name_window = "harris_corner_trackbar"

cv2.namedWindow(name_window)
cv2.createTrackbar("threshold", name_window, 125, 255, lambda value: None)
cv2.createTrackbar("block", name_window, 2, 10, lambda value: None)
cv2.createTrackbar("ksize", name_window, 0, 2, lambda value: None)
cv2.createTrackbar("k", name_window, 40, 100, lambda value: None)

while True:
    threshold = cv2.getTrackbarPos("threshold", name_window)
    block_size = max(2, cv2.getTrackbarPos("block", name_window))
    ksize = cv2.getTrackbarPos("ksize", name_window) * 2 + 3
    k_value = max(1, cv2.getTrackbarPos("k", name_window)) / 1000

    response = cv2.cornerHarris(src=gray_float, blockSize=block_size,
                                ksize=ksize, k=k_value)
    response = cv2.dilate(src=response, kernel=None)

    response_norm = cv2.normalize(src=response, dst=None, alpha=0, beta=255,
                                  norm_type=cv2.NORM_MINMAX)
    response_norm = np.uint8(response_norm)

    _, corner_mask = cv2.threshold(src=response_norm, thresh=threshold,
                                   maxval=255, type=cv2.THRESH_BINARY)
    _, _, _, centers = cv2.connectedComponentsWithStats(image=corner_mask)

    result = img.copy()
    for center in centers[1:]:
        x, y = np.round(center).astype(np.int32)
        cv2.circle(img=result, center=(int(x), int(y)), radius=4,
                   color=(0, 0, 255), thickness=-1)

    header = np.full(shape=(48, result.shape[1], 3), fill_value=255,
                     dtype=np.uint8)
    text = (f"threshold={threshold}, block={block_size}, "
            f"ksize={ksize}, k={k_value:.3f}, "
            f"corners={max(0, len(centers) - 1)}")
    cv2.putText(img=header, text=text, org=(10, 31),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    show = np.vstack(tup=[header, result])

    cv2.imshow(name_window, show)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output, "04_harris_trackbar.png")
        cv2.imwrite(filename=path_save, img=show)
        print("Saved:", path_save)

cv2.destroyAllWindows()
