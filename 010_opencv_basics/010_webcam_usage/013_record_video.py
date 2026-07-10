import cv2
import os
from datetime import datetime

dir_proj = os.getcwd()
dir_output = os.path.join(dir_proj, "010_opencv_basics",
                          "010_webcam_usage", "output", "video")
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


"""===========================================================================================
Capture Webcam Frames as a Video File
==========================================================================================="""
# Initialize the video writer and recording status.
video_writer = None
is_recording = False
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("mywindow", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    # Start or stop recording when the 'r' key is pressed.
    if key == ord('r'):

        # If recoding is not currently active, start recording.
        if not is_recording:
            # Generate a timestamp string for the video filename.
            time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Define the file path for saving the recorded video.
            file_path = os.path.join(dir_output, f"video_{time_now}.mp4")

            # Create a VideoWriter instance for writing frames to the video file.
            video_writer = cv2.VideoWriter(
                filename=file_path,
                fourcc=cv2.VideoWriter_fourcc(*"MJPG"),
                fps=cap.get(cv2.CAP_PROP_FPS),
                frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            )

            # Check whether the video file was created successfully.
            if video_writer.isOpened():
                is_recording = True
                print(f"Recording started: {file_path}")
            else:
                print("Error: Could not create video file")

        # If recoding is currently active, stop recording.
        else:
            # Update the recording status and release the video writer.
            is_recording = False
            video_writer.release()
            video_writer = None
            print("Recording stopped")

    # If recoding is currently acrive, write the current frame to the video file.
    if is_recording:
        video_writer.write(frame)

cap.release()
cv2.destroyAllWindows()
