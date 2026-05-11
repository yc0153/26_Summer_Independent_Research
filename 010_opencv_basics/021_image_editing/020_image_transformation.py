import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "010_opencv_basics",
                        "021_image_editing")
img_path = os.path.join(dir_base, "input", "sample.png")
dir_output = os.path.join(dir_base, "output", "transform_image")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(img_path)
if img is None:
    print("Error: Could not open image")
    exit()

# Store the size of the loaded image.
height, width = img.shape[:2]

"""===========================================================================================
Simple Transformation Using OpenCV Built-in Functions
==========================================================================================="""
# Rotate the image 90 degrees clockwise.
rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite(os.path.join(dir_output, "img_rotated_90.png"), rotated_90)

# Flip the image horizontally.
flipped = cv2.flip(img, 1)
cv2.imwrite(os.path.join(dir_output, "img_flipped.png"), flipped)

# Scale the image down by a factor of 0.5
scaled = cv2.resize(img, None, fx=0.5, fy=0.5)
cv2.imwrite(os.path.join(dir_output, "img_scaled.png"), scaled)


"""===========================================================================================
Transformation Using a Matrix: Example of Translation
==========================================================================================="""
# Define translation matrix.
trans_matrix = np.float32([[1, 0, 120],
                           [0, 1, 80],
                           [0, 0, 1]])

# Translate the image.
translated = cv2.warpPerspective(img, trans_matrix, (width, height))

# Save the result.
cv2.imwrite(os.path.join(dir_output, "image_translated.png"), translated)

"""===========================================================================================
Transformation Using a Matrix: Example of Shearing
==========================================================================================="""
# Define the shearing matrix.
trans_matrix = np.float32([[1, 0.25, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

# Shear the image.
sheared = cv2.warpPerspective(
    img, trans_matrix, (int(width + height * 0.25), height))

# Save the result.
cv2.imwrite(os.path.join(dir_output, "image_sheared.png"), sheared)

"""===========================================================================================
Transformation Using a Matrix: Example of Reflection
==========================================================================================="""
# Define the reflection matrix.
trans_matrix = np.float32([[-1, 0, width - 1],
                           [0, 1, 0],
                           [0, 0, 1]])

# Reflect the image horizontally.
flipped_by_matrix = cv2.warpPerspective(img, trans_matrix, (width, height))

# Save the result.
cv2.imwrite(os.path.join(dir_output, "image_flipped.png"),
            flipped_by_matrix)

"""===========================================================================================
Transformation Using a Matrix: Example of Scailing
==========================================================================================="""
# Define the scailing factor.
scale = 0.5

# Define the scailing matrix.
trans_matrix = np.float32([[scale, 0, 0],
                           [0, scale, 0],
                           [0, 0, 1]])

# Scale the image.
scaled_by_matrix = cv2.warpPerspective(
    img, trans_matrix, (int(width * scale), int(height * scale)))

# Define the four source corner points
cv2.imwrite(os.path.join(dir_output, "image_scaled.png"), scaled_by_matrix)


"""===========================================================================================
Transformation Using Matrices: Example of Rotation About the Image Center
==========================================================================================="""
# Convert the rotation angle from degrees to radians.
angle = np.deg2rad(30)

# Get the center of the image.
cx, cy = width / 2, height / 2

# Define the transformation matrices.
move_to_origin = np.float32([[1, 0, -cx],
                             [0, 1, -cy],
                             [0, 0, 1]])

rotate = np.float32([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

move_back = np.float32([[1, 0, cx],
                        [0, 1, cy],
                        [0, 0, 1]])

# Combine the matrices to create the full transformation matrix.
trans_matrix = move_back @ rotate @ move_to_origin

# Apply the rotation to the image.
img_rotated = cv2.warpPerspective(
    src=img, M=trans_matrix, dsize=(width, height))

# Save the result.
cv2.imwrite(os.path.join(dir_output, "img_rotated_30.png"), img_rotated)

"""===========================================================================================
Transformation Using a Matrix: Example of Perspective Transformation with Four Corner Points
==========================================================================================="""
# Define the four source corner points.
src = np.float32([[0, 0],
                  [width - 1, 0],
                  [0, height - 1],
                  [width - 1, height - 1]])

# Define the four destination corner points.
dst = np.float32([[80, 40],
                  [width - 120, 0],
                  [0, height - 60],
                  [width - 1, height - 1]])

# Compute the perspective transformation matrix.
trans_matrix = cv2.getPerspectiveTransform(src, dst)

# Apply the perspective transformation to the image.
perspective = cv2.warpPerspective(img, trans_matrix, (width, height))

# Save the result
cv2.imwrite(os.path.join(dir_output, "image_perspective.png"), perspective)
