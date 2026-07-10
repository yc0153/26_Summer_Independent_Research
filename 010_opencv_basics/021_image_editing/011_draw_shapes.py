import os
import cv2
import numpy as np

dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "010_opencv_basics",
                        "021_image_editing")
dir_output = os.path.join(dir_base, "output", "draw_shapes")
os.makedirs(dir_output, exist_ok=True)

"""===========================================================================================
Drawing Shapes
==========================================================================================="""
# Create a white background image.
img = np.full((500, 700, 3), 255, dtype=np.uint8)

# Drawn a line.
cv2.line(
    img=img,
    pt1=(50, 80),
    pt2=(650, 80),
    color=(255, 0, 0),
    thickness=5
)

# Drawn a rectangle.
cv2.rectangle(
    img=img,
    pt1=(80, 150),
    pt2=(280, 350),
    color=(0, 255, 0),
    thickness=4
)

# Drawn a circle.
cv2.circle(
    img=img,
    center=(500, 250),
    radius=100,
    color=(0, 0, 255),
    thickness=4
)

# Save the resulting image to a file.
cv2.imwrite(os.path.join(dir_output, "draw_shapes.jpg"), img)
