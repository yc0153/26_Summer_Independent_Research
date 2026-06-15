import os
import shutil
import cv2
import numpy as np


dir_proj = os.getcwd()
dir_base = os.path.join(dir_proj, "001_camera_calibration")
dir_input = os.path.join(dir_base, "input")
dir_output = os.path.join(dir_base, "output")
dir_capture = os.path.join(dir_input, "calibration_captures")
os.makedirs(dir_output, exist_ok=True)

if os.path.exists(dir_capture):
    shutil.rmtree(path=dir_capture)
os.makedirs(dir_capture, exist_ok=True)


"""===========================================================================================
Camera Calibration Settings
==========================================================================================="""
camera_index = 0
board_inner_corners = (10, 8)
square_size_mm = 5.5
target_capture_count = 30
path_calibration = os.path.join(dir_output, "camera_calibration.yml")


"""===========================================================================================
Prepare Calibration Data
==========================================================================================="""
cols, rows = board_inner_corners
board_points = np.zeros(shape=(cols * rows, 3), dtype=np.float32)
board_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
board_points *= square_size_mm

object_points = []
image_points = []
image_size = None

calibrated = False
show_undistorted = False
extrinsic_index = 0

rms_error = None
mean_error = None
camera_matrix = None
new_camera_matrix = None
distortion_coeffs = None
rvecs = None
tvecs = None
extrinsic_matrices = []


"""===========================================================================================
Open Camera
==========================================================================================="""
cap = cv2.VideoCapture(camera_index)
cv2.namedWindow(winname="camera_calibration", flags=cv2.WINDOW_NORMAL)


"""===========================================================================================
Capture, Calibrate, and Show Result
==========================================================================================="""
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(image=gray,
                                               patternSize=board_inner_corners,
                                               flags=flags)

    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)
        corners = cv2.cornerSubPix(image=gray, corners=corners,
                                   winSize=(11, 11), zeroZone=(-1, -1),
                                   criteria=criteria)

    preview = frame.copy()

    if found:
        cv2.drawChessboardCorners(image=preview,
                                  patternSize=board_inner_corners,
                                  corners=corners,
                                  patternWasFound=found)

    if calibrated and show_undistorted:
        preview = cv2.undistort(src=preview,
                                cameraMatrix=camera_matrix,
                                distCoeffs=distortion_coeffs,
                                newCameraMatrix=new_camera_matrix)

    header = np.full(shape=(86, preview.shape[1], 3), fill_value=255,
                     dtype=np.uint8)
    board_text = "found" if found else "not found"
    mode_text = "undistorted" if show_undistorted else "raw"
    state_text = "calibrated" if calibrated else "capture"

    if len(image_points) >= target_capture_count and not calibrated:
        state_text = "press k"

    cv2.putText(img=header,
                text="c: capture | k: calibrate | r: toggle | q: quit",
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.58,
                color=(0, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA)
    cv2.putText(img=header,
                text=(f"samples {len(image_points)}/{target_capture_count} | "
                      f"board {board_text} | mode {mode_text} | "
                      f"{state_text}"),
                org=(10, 62),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.52,
                color=(0, 0, 180),
                thickness=2,
                lineType=cv2.LINE_AA)

    camera_view = np.vstack(tup=[header, preview])
    cv2.imshow(winname="camera_calibration", mat=camera_view)

    if calibrated:
        result_view = np.full(shape=(720, 1100, 3), fill_value=255,
                              dtype=np.uint8)
        distortion_values = distortion_coeffs.ravel()
        extrinsic_matrix = extrinsic_matrices[extrinsic_index]

        result_lines = [
            "Calibration Result",
            f"samples: {len(extrinsic_matrices)}",
            f"rms error: {rms_error:.6f}",
            f"mean reprojection error: {mean_error:.6f}",
            "",
            "Intrinsic matrix K:",
        ]

        for row in camera_matrix:
            values = [f"{value:11.4f}" for value in row]
            result_lines.append("[" + " ".join(values) + "]")

        result_lines += [
            "",
            "Distortion coefficients:",
            "[" + " ".join([f"{value:.6f}"
                             for value in distortion_values]) + "]",
            "",
            (f"Extrinsic matrix [R | t] "
             f"{extrinsic_index + 1}/{len(extrinsic_matrices)}:"),
        ]

        for row in extrinsic_matrix:
            values = [f"{value:11.4f}" for value in row]
            result_lines.append("[" + " ".join(values) + "]")

        result_lines += [
            "",
            "n: next extrinsic | p: previous extrinsic | q: quit",
        ]

        y = 36
        for line in result_lines:
            color = (0, 0, 180) if "matrix" in line.lower() else (0, 0, 0)
            cv2.putText(img=result_view,
                        text=line,
                        org=(24, y),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.62,
                        color=color,
                        thickness=2,
                        lineType=cv2.LINE_AA)
            y += 30

        cv2.imshow(winname="calibration_result", mat=result_view)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("c") and found and not calibrated:
        object_points.append(board_points.copy())
        image_points.append(corners.copy())
        image_size = (frame.shape[1], frame.shape[0])

        capture_count = len(image_points)
        path_capture = os.path.join(dir_capture,
                                    f"capture_{capture_count:02d}.png")
        cv2.imwrite(filename=path_capture, img=frame)

    elif (key == ord("k")
          and len(image_points) >= target_capture_count
          and not calibrated):
        rms_error, camera_matrix, distortion_coeffs, rvecs, tvecs = (
            cv2.calibrateCamera(objectPoints=object_points,
                                imagePoints=image_points,
                                imageSize=image_size,
                                cameraMatrix=None,
                                distCoeffs=None)
        )

        total_error = 0
        extrinsic_matrices = []

        for obj_points, img_points, rvec, tvec in zip(object_points,
                                                     image_points,
                                                     rvecs,
                                                     tvecs):
            projected_points, _ = cv2.projectPoints(
                objectPoints=obj_points,
                rvec=rvec,
                tvec=tvec,
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coeffs)
            error = cv2.norm(src1=img_points,
                             src2=projected_points,
                             normType=cv2.NORM_L2) / len(projected_points)
            total_error += error

            rotation_matrix, _ = cv2.Rodrigues(src=rvec)
            extrinsic_matrix = np.hstack(tup=[rotation_matrix, tvec])
            extrinsic_matrices.append(extrinsic_matrix)

        mean_error = total_error / len(object_points)
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coeffs,
            imageSize=image_size,
            alpha=1,
            newImgSize=image_size)

        storage = cv2.FileStorage(path_calibration, cv2.FILE_STORAGE_WRITE)
        storage.write("board_inner_corners_width", board_inner_corners[0])
        storage.write("board_inner_corners_height", board_inner_corners[1])
        storage.write("square_size_mm", square_size_mm)
        storage.write("capture_count", len(extrinsic_matrices))
        storage.write("rms_error", float(rms_error))
        storage.write("mean_reprojection_error", float(mean_error))
        storage.write("camera_matrix", camera_matrix)
        storage.write("distortion_coeffs", distortion_coeffs)

        for index, (rvec, tvec, extrinsic_matrix) in enumerate(
                zip(rvecs, tvecs, extrinsic_matrices), start=1):
            storage.write(f"rvec_{index:02d}", rvec)
            storage.write(f"tvec_{index:02d}", tvec)
            storage.write(f"extrinsic_matrix_{index:02d}", extrinsic_matrix)

        storage.release()

        cv2.namedWindow(winname="calibration_result",
                        flags=cv2.WINDOW_NORMAL)
        calibrated = True
        extrinsic_index = 0

    elif key == ord("r") and calibrated:
        show_undistorted = not show_undistorted

    elif key == ord("n") and calibrated:
        extrinsic_index = (extrinsic_index + 1) % len(extrinsic_matrices)

    elif key == ord("p") and calibrated:
        extrinsic_index = (extrinsic_index - 1) % len(extrinsic_matrices)


"""===========================================================================================
Close Camera
==========================================================================================="""
cap.release()
cv2.destroyAllWindows()
