import csv
import os
import depthai as dai
import numpy as np

def get_oak_camera_intrinsics():
    """
    Retrieves the camera intrinsics and distortion coefficients from an OAK device.

    Returns:
        camera_intrinsics_matrix (np.ndarray): 3x3 camera intrinsics matrix.
        distortion_coeffs (np.ndarray): 1x5 distortion coefficients array.
    """
    with dai.Device() as device:
        calib_data = device.readCalibration()
        # Replace CAM_A with the correct camera board socket (e.g., CAM_C for right, CAM_B for left, CAM_A for RGB)
        cam_socket = dai.CameraBoardSocket.CAM_A
        camera_intrinsics_matrix = np.array(calib_data.getCameraIntrinsics(cam_socket))
        distortion_coeffs = np.array(calib_data.getDistortionCoefficients(cam_socket)[:5]).reshape((1,5))
        camera_intrinsics_matrix, distortion_coeffs = get_oak_camera_intrinsics(device=device)
        print("Camera Intrinsics Matrix:\n", camera_intrinsics_matrix)
        print("Distortion Coefficients:\n", distortion_coeffs)
        
if __name__ == "__main__":
    get_oak_camera_intrinsics()