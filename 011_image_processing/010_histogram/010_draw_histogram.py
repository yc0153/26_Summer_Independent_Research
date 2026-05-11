import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "010_histogram")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(filename=os.path.join(dir_input, "flower.png"))
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Draw Brightness Histogram
==========================================================================================="""


def draw_histogram(gray):
    # Count how many pixels belong to each brightness value from 0 to 255.
    hist = cv2.calcHist(images=[gray], channels=[0], mask=None,
                        histSize=[256], ranges=[0, 256])
    hist = cv2.normalize(src=hist, dst=None, alpha=0, beta=260,
                         norm_type=cv2.NORM_MINMAX)

    canvas = np.full(shape=(300, 256, 3), fill_value=255, dtype=np.uint8)
    for i in range(1, 256):
        x1, x2 = (i - 1), i
        y1 = 300 - int(hist[i - 1][0])
        y2 = 300 - int(hist[i][0])
        cv2.line(img=canvas, pt1=(x1, y1), pt2=(x2, y2),
                 color=(30, 30, 30), thickness=2)

    return canvas


"""===========================================================================================
Compare Brightness Distributions
==========================================================================================="""
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
dark = cv2.convertScaleAbs(src=img, alpha=0.55, beta=-35)
bright = cv2.convertScaleAbs(src=img, alpha=1.15, beta=40)

gray_dark = cv2.cvtColor(src=dark, code=cv2.COLOR_BGR2GRAY)
gray_bright = cv2.cvtColor(src=bright, code=cv2.COLOR_BGR2GRAY)

cv2.imwrite(filename=os.path.join(dir_output, "01_gray.png"), img=gray)
cv2.imwrite(filename=os.path.join(dir_output, "02_hist_gray.png"),
            img=draw_histogram(gray=gray))

cv2.imwrite(filename=os.path.join(dir_output, "03_dark.png"), img=dark)
cv2.imwrite(filename=os.path.join(dir_output, "04_hist_dark.png"),
            img=draw_histogram(gray=gray_dark))

cv2.imwrite(filename=os.path.join(dir_output, "05_bright.png"), img=bright)
cv2.imwrite(filename=os.path.join(dir_output, "06_hist_bright.png"),
            img=draw_histogram(gray=gray_bright))

print("Saved output images to:", dir_output)
