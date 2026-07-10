import os
import cv2

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "011_image_processing", "060_edges")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(filename=os.path.join(dir_input, "flower.png"))
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Canny Edge Detection
==========================================================================================="""
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)

# Canny uses low/high thresholds to keep connected edge lines.
canny = cv2.Canny(image=blur, threshold1=50, threshold2=150)

cv2.imwrite(filename=os.path.join(dir_output, "01_gray.png"), img=gray)
cv2.imwrite(filename=os.path.join(dir_output, "03_canny.png"), img=canny)

print("Saved output images to:", dir_output)
