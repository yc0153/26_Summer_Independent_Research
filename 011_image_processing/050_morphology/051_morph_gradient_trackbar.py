import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "050_morphology")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_input, exist_ok=True)
os.makedirs(dir_output, exist_ok=True)

path_input = os.path.join(dir_input, "mask_noisy.png")


"""===========================================================================================
Create Binary Mask Input
==========================================================================================="""
if not os.path.exists(path_input):
    mask = np.zeros(shape=(320, 520), dtype=np.uint8)
    cv2.rectangle(img=mask, pt1=(80, 80), pt2=(240, 230),
                  color=255, thickness=-1)
    cv2.circle(img=mask, center=(365, 155), radius=80,
               color=255, thickness=-1)
    cv2.circle(img=mask, center=(365, 155), radius=28,
               color=0, thickness=-1)

    rng = np.random.default_rng(seed=5)
    for x, y in rng.integers(low=[0, 0], high=[520, 320],
                             size=(120, 2)):
        mask[y, x] = 255

    cv2.imwrite(filename=path_input, img=mask)

mask = cv2.imread(filename=path_input, flags=cv2.IMREAD_GRAYSCALE)
if mask is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Helper Function
==========================================================================================="""
def add_label(gray, text):
    result = cv2.cvtColor(src=gray, code=cv2.COLOR_GRAY2BGR)
    cv2.putText(img=result, text=text, org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    return result


"""===========================================================================================
Morphological Gradient Trackbar
==========================================================================================="""
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE,
                                   ksize=(3, 3))
name_window = "morph_gradient_iterations"

cv2.namedWindow(name_window)
cv2.createTrackbar("iterations", name_window, 1, 10, lambda value: None)

while True:
    iterations = max(1, cv2.getTrackbarPos("iterations", name_window))
    result = cv2.morphologyEx(src=mask, op=cv2.MORPH_GRADIENT,
                              kernel=kernel, iterations=iterations)

    preview = np.hstack(tup=[
        add_label(gray=mask, text="input"),
        add_label(gray=result, text=f"gradient x{iterations}")
    ])

    cv2.imshow(name_window, preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("s"):
        path_save = os.path.join(dir_output,
                                 "11_morph_gradient_trackbar.png")
        cv2.imwrite(filename=path_save, img=preview)
        print("Saved:", path_save)

cv2.destroyAllWindows()
