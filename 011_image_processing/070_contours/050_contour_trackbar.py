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

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)


"""===========================================================================================
Contour Trackbar
==========================================================================================="""
name_window = "contour_trackbar"

# min_area slider is multiplied by 100.
# epsilon slider means percent of contour perimeter.
cv2.namedWindow(name_window)
cv2.createTrackbar("threshold", name_window, 120, 255, lambda value: None)
cv2.createTrackbar("min_area", name_window, 10, 80, lambda value: None)
cv2.createTrackbar("epsilon", name_window, 3, 15, lambda value: None)

while True:
    threshold = cv2.getTrackbarPos("threshold", name_window)
    min_area = cv2.getTrackbarPos("min_area", name_window) * 100
    epsilon_percent = cv2.getTrackbarPos("epsilon", name_window)

    _, binary = cv2.threshold(src=gray, thresh=threshold, maxval=255,
                              type=cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        image=binary, mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()
    kept = 0

    for contour in contours:
        area = cv2.contourArea(contour=contour)
        if area < min_area:
            continue

        kept += 1
        perimeter = cv2.arcLength(curve=contour, closed=True)
        approx = cv2.approxPolyDP(
            curve=contour,
            epsilon=(epsilon_percent / 100.0) * perimeter,
            closed=True)
        x, y, w, h = cv2.boundingRect(array=contour)

        cv2.drawContours(image=result, contours=[approx], contourIdx=-1,
                         color=(0, 0, 255), thickness=2)
        cv2.rectangle(img=result, pt1=(x, y), pt2=(x + w, y + h),
                      color=(255, 0, 0), thickness=2)
        cv2.putText(img=result, text=f"v={len(approx)}", org=(x, y - 8),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                    color=(0, 0, 255), thickness=2)

    header = np.full(shape=(56, result.shape[1], 3),
                     fill_value=255, dtype=np.uint8)
    cv2.putText(img=header, text=f"T={threshold} area={min_area}",
                org=(10, 22), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(img=header, text=f"eps={epsilon_percent}% count={kept}",
                org=(10, 46), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA)
    preview = np.vstack(tup=[header, result])

    cv2.imshow(name_window, preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output, "07_contour_trackbar.png")
        cv2.imwrite(filename=path_save, img=preview)
        print("Saved:", path_save)

cv2.destroyAllWindows()
