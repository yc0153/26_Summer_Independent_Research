"""===========================================================================================
Required Libraries
==========================================================================================="""
import cv2

# Import the built-in libraries for definiing save directories and generating timestamps.
import os
from datetime import datetime

"""===========================================================================================
Directory Setting for saving the photos that will be taken
==========================================================================================="""
# Create the output directory for saving captured photos.
dir_proj = os.getcwd()
dir_output = os.path.join(dir_proj, "010_opencv_basics",
                          "010_webcam_usage", "output", "photo")
os.makedirs(name=dir_output, exist_ok=True)

cap = cv2.VideoCapture(1)
if cap.isOpened():
    print("camera opened successfully")
else:
    print("Error: Could not open camera")
    exit()

target_width = 1280
target_height = 720
target_fps = 100
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
cap.set(cv2.CAP_PROP_FPS, target_fps)
print("Frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frames per second:", cap.get(cv2.CAP_PROP_FPS))

cv2.namedWindow("mywindow", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("mywindow", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('c'):
        """===========================================================================================
        Capute the Current Webcam Frame and Save It in the Predefined Directory
        ==========================================================================================="""
        # Get the current time and format it as a timestamp string.
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Define the file fath for saving the captured photo using the current timestamp.
        file_path = os.path.join(
            dir_output, f"photo_{time_now}.jpg")
        
        # Save the current frame as an image file.
        cv2.imwrite(file_path, frame)

        # Print the save result.
        print(f"Photo saved at: {file_path}")

cap.release()
cv2.destroyAllWindows()
