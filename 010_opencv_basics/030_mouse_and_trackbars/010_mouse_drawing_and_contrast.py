import os
import cv2

dir_proj = os.getcwd()
dir_input = os.path.join(dir_proj, "010_opencv_basics",
                         "030_mouse_and_trackbars", "input")
dir_output = os.path.join(dir_proj, "010_opencv_basics",
                          "030_mouse_and_trackbars", "output")
os.makedirs(dir_output, exist_ok=True)

img = cv2.imread(os.path.join(dir_input, "sample.png"))
if img is None:
    print("Error: Could not open image")
    exit()

"""===========================================================================================
Mouse Drawing
==========================================================================================="""
# Set the window name.
name_window = "mouse_drawing"

# Create a copy of the original image for drawing.
img_drawing = img.copy()

# Initialize drawing state variables.
is_drawing = False
prev_point = None


def do_nothing_function(value):
    # /// Dummy callback function for trackbars ///
    pass


def mouse_callback(event, x, y, flags, param):
    # /// Mouse handling function ///
    global is_drawing, prev_point

    # Start drawing when the left mouse button is pressed.
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        prev_point = (x, y)

    # Draw lines while the mouse moves with the left button held down.
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        cv2.line(img_drawing, prev_point, (x, y), (0, 0, 255), 3)
        prev_point = (x, y)

    # Stop drawing when the left mouse button is released.
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False


# Create the window
cv2.namedWindow(name_window)

# Register the mouse callback for the window.
cv2.setMouseCallback(name_window, mouse_callback)

# Create a contrast trackbar.
cv2.createTrackbar("contrast", name_window, 100, 300, do_nothing_function)

# Create a brightness trackbar.
cv2.createTrackbar("brightness", name_window, 100, 200, do_nothing_function)

# Main drawing loop.
while True:
    # Read trackbar values and apply contrast and brightness to the drawing.
    contrast = cv2.getTrackbarPos("contrast", name_window) / 100
    brightness = cv2.getTrackbarPos("brightness", name_window) - 100
    img_modified = cv2.convertScaleAbs(
        img_drawing, alpha=contrast, beta=brightness)

    cv2.imshow(name_window, img_modified)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # Reset the drawing when 'r' is pressed.
    elif key == ord("r"):
        img_drawing = img.copy()

    # Save the drawing when 's' is pressed.
    elif key == ord("s"):
        file_path = os.path.join(dir_output, "mouse_drawing_result.png")
        cv2.imwrite(file_path, img_modified)
        print("Saved:", file_path)

cv2.destroyAllWindows()
