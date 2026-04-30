import cv2

# Create a VideoCapture object for the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if cap.isOpened():
    print("camera opened successfully")
else:
    print("Error: Could not open camera")
    exit()

# Display the camera feed
cv2.namedWindow("window_1", cv2.WINDOW_NORMAL)

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
