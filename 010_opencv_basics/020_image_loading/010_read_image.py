import os
import cv2

"""===========================================================================================
Set Directories
==========================================================================================="""
dir_proj = os.getcwd()
img_namge = 'sample.png'
img_path = os.path.join(dir_proj, "010_opencv_basics",
                        "020_image_loading", "input", img_namge)

"""===========================================================================================
Read Image
==========================================================================================="""
# Read the image with its original channels and stop if loading fails.
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print("Error: Could not open image")
    exit()


"""===========================================================================================
Inspect the Loaded Image
==========================================================================================="""
# Print the height, width, and number of channels of the loaded image.
height, width = img.shape[:2]
channels = 1 if len(img.shape) == 2 else img.shape[2]
print("Image path:", img_path)
print("Width:", width)
print("Height:", height)
print("Channels:", channels)

# Print the data type and total pixels of the loaded image.
print("Data type:", img.dtype)
print("Total pixels:", img.size)

# Print a small region of pixel intensities for each channel.
pt_left_top = [200, 205]
pt_right_bot = [200, 205]
if channels == 4:
    b, g, r, a = cv2.split(img)
    print("Partial Blue channel:\n", b[pt_left_top[0]:pt_left_top[1],
                                       pt_right_bot[0]:pt_right_bot[1]])
    print("Partial Green channel:\n", g[pt_left_top[0]:pt_left_top[1],
                                        pt_right_bot[0]:pt_right_bot[1]])
    print("Partial Red channel:\n", r[pt_left_top[0]:pt_left_top[1],
                                      pt_right_bot[0]:pt_right_bot[1]])
    print("Partial Alpha channel:\n", a[pt_left_top[0]:pt_left_top[1],
                                        pt_right_bot[0]:pt_right_bot[1]])
    print("Alpha channel min value:", a.min())
    print("Alpha channel max value:", a.max())
elif channels == 3:
    b, g, r = cv2.split(img)
    print("Partial Blue channel:\n", b[pt_left_top[0]:pt_left_top[1],
                                       pt_right_bot[0]:pt_right_bot[1]])
    print("Partial Green channel:\n", g[pt_left_top[0]:pt_left_top[1],
                                        pt_right_bot[0]:pt_right_bot[1]])
    print("Partial Red channel:\n", r[pt_left_top[0]:pt_left_top[1],
                                      pt_right_bot[0]:pt_right_bot[1]])
else:
    print("Partial Gray channel:\n", img[pt_left_top[0]:pt_left_top[1],
                                         pt_right_bot[0]:pt_right_bot[1]])

# Show the loaded image and wait for a key press.
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
