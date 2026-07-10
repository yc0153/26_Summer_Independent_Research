"""===========================================================================================
Required Libraries
==========================================================================================="""
# Import the OpenCV library.
import cv2

"""===========================================================================================
Preparation for Capturing
==========================================================================================="""
# Create a VideoCapture instance named 'cap' that uses camera index 1.
cap = cv2.VideoCapture(1)

# Check whther the 'cap' instance is connected to the camera properly.
if cap.isOpened():
    print("Camera opened successfully")
else:
    print("Error: Could not open camera")
    exit()

# Create a named window with a window featrue flag.
cv2.namedWindow(winname="mywindow", flags=cv2.WINDOW_NORMAL)


"""===========================================================================================
Iteratively Read the Camera Frames and Display Them in the Created Named Window
==========================================================================================="""
while True:
    # Read a frame from the 'cap' instance.
    # The result is returned as a tuple: (ret, frame)
    # 'ret' refers to the return status. It returns a Boolean value that indicates whether the frame was read successfully.
    ret, frame = cap.read()

    # Check whether the frame was read successfully.
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame in the predefined named window.
    cv2.imshow("mywindow", frame)

    # Check the termination condition by detecting whether the 'q' key is pressed.
    # cv2.waitKey takes an integer value that represents the delay in milliseconds for waiting for a key input.
    # If the input is 0, it waits indefinitely until a key is pressed.
    # 0xFF is a hexadecimal value equal to 255 in decimal. 
    # Therefore, cv2.waitKey(1) & 0xFF ensures that only the last 8 bits of the pressed key value are used.
    # The ord() function returns the Unicode of the input character.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""===========================================================================================
After using the camera
==========================================================================================="""
# Release the camera resource.
# It means returning the resouces used during capturing, such as the camera, memory, and windows, to the operating system.
cap.release()

# Close the named window.
cv2.destroyWindow(winname="mywindow")