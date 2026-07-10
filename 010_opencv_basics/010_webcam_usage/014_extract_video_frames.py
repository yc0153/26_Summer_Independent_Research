import os
import cv2

"""===========================================================================================
Define the Save Directory
==========================================================================================="""
dir_proj = os.getcwd()
video_name = "video_2026-05-10_18-33-19"
video_name_full = "video_2026-05-10_18-33-19" + ".mp4"
path_video = os.path.join(dir_proj, "010_opencv_basics",
                          "010_webcam_usage", "output", "video", video_name_full)
dir_output = os.path.join(dir_proj, "010_opencv_basics",
                          "010_webcam_usage", "output", "extracted_frames", video_name)
os.makedirs(name=dir_output, exist_ok=True)

"""===========================================================================================
Extract Video Frames from the Saved Video File
==========================================================================================="""
# Open the video file as a VideoCapture instance.
cap = cv2.VideoCapture(path_video)

# Initialize the frame index.
frame_idx = 0

while True:
    # Read the next frame from the video file.
    ret, frame = cap.read()
    
    # Stop the loop when there are no more frames to read.
    if not ret:
        break

    else:
        # Define the image file path for the current frame.
        file_path = os.path.join(dir_output, f"frame_{frame_idx:06d}.jpg")

        # Save current frame as an image file.
        cv2.imwrite(file_path, frame)

        # Increment the frame index.
        frame_idx += 1

# Release the video file resource.
cap.release()

# Print a completion message
print(f"Extraction completed.")