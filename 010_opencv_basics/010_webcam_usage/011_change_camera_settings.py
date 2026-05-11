import cv2

"""===========================================================================================
Reference Data for Selecting Webcam Resolutions
==========================================================================================="""
# Define common resolution names and sizes for comparison.
resolutions = {
    # 4:3, Quarter Video Graphics Array, Very low resolution
    "QVGA": (320, 240),
    # 4:3, Video Graphics Array, Basic resolution for webcams
    "VGA": (640, 480),
    # 4:3, Super Video Graphics Array, Older standard for computer monitors
    "SVGA": (800, 600),
    # 4:3, eXtended Graphics Array, Medium resolution for computer monitors
    "XGA": (1024, 768),
    # 16:9, 720p, High Definition, Basic resolution for modern webcams and streaming
    "HD": (1280, 720),
    # 16:9, 1080p, Full High Definition, Common resolution for webcams and streaming
    "Full HD": (1920, 1080),
    # 16:9, 1440p, Quad High Definition, Higher resolution for high-end webcams and monitors
    "QHD": (2560, 1440),
    # 16:9, 4K, Ultra High Definition, High-end resolution for professional cameras and monitors
    "UHD": (3840, 2160),
    # 16:9, 8K Resolution, Extremely high resolution for professional cameras and monitors
    "8K": (7680, 4320)
}

cap = cv2.VideoCapture(1)
if cap.isOpened():
    print("camera opened successfully")
else:
    print("Error: Could not open camera")
    exit()


"""===========================================================================================
Change the Camera Settings
==========================================================================================="""
# Print the default settings of the conncected camera.
print("Default frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Default frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Default frames per second:", cap.get(cv2.CAP_PROP_FPS))

# Change the target FPS, frame width, and frame height to the camera
resolution_name = ["QVGA", "VGA", "SVGA", "XGA",
                   "HD", "Full HD", "QHD", "UHD", "8K"][7]
target_fps = 100
target_width = resolutions[resolution_name][0]
target_height = resolutions[resolution_name][1]

# Apply the target FPS, frame width, and frame height to the camera
cap.set(cv2.CAP_PROP_FPS, target_fps)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)

# Print the updated camera settings.
print("-> Modified frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("-> Modified frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("-> Modified frames per second:", cap.get(cv2.CAP_PROP_FPS))

cv2.namedWindow("mywindow", cv2.WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("mywindow", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
