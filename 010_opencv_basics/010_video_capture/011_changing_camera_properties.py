import cv2

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

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if cap.isOpened():
    print("camera opened successfully")
    print("Default frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Default frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Default frames per second:", cap.get(cv2.CAP_PROP_FPS))
else:
    print("Error: Could not open camera")
    exit()

# Set new properties for the camera
resolution_name = ["QVGA", "VGA", "SVGA",
                   "XGA", "HD", "Full HD",
                   "QHD", "UHD", "8K"][4]
target_width = resolutions[resolution_name][0]
target_height = resolutions[resolution_name][1]
target_fps = 10

cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
cap.set(cv2.CAP_PROP_FPS, target_fps)

# Check the changed properties
print("==========================")
print("Modified frame width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Modified frame height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Modified frames per second:", cap.get(cv2.CAP_PROP_FPS))

# Display the camera feed
cv2.namedWindow("window_1", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window_1", int(target_width), int(target_height))

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("window_1", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        print("Failed to grab frame")
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
