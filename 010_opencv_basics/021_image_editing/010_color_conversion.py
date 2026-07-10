import os
import cv2

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "010_opencv_basics",
                        "021_image_editing")
img_path = os.path.join(dir_base, "input", "sample.png")
dir_output = os.path.join(dir_base, "output", "color_conversion")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(img_path)
if img is None:
    print("Error: Could not open image")
    exit()

"""===========================================================================================
Convert Colors
==========================================================================================="""
# Convert the BGR image to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(dir_output, "color_01_gray.png"), gray)

# Saving an RGB image directly produces incorrect colors because cv2.imwrite expects BGR.
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(dir_output, "color_03_rgb_saved_wrong.png"), rgb)

# Convert RGB back to BGR before saving to keep the original colors.
rgb_saved_correctly = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(dir_output, "color_02_rgb_correct.png"),
            rgb_saved_correctly)

# Change the Hue channel in HSV space to shift the overall color tone.
# Visit https://colorizer.org/ to experiment with HSV values.
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv[:, :, 0] = (hsv[:, :, 0] + 60) % 180
hue_changed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite(os.path.join(dir_output, "color_04_hue_changed.png"), hue_changed)

# Invert colors by subtracting each pixel value from 255.
inverted = 255 - img
cv2.imwrite(os.path.join(dir_output, "color_05_inverted.png"), inverted)