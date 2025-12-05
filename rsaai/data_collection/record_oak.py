
#!/usr/bin/env python3

"""
Recording utility for OAK-D main RGB camera, depth, and IMU.

Display: Only the main ISP stream window is shown.

Controls:
    - 'r' : Start recording
    - 's' : Stop recording
    - 'q' : Quit
    - '/' : Toggle printing camera settings (exposure, ISO, lens, temperature)
    - Optional manual camera controls preserved: 't' AF trigger, 'f' AF continuous,
      'e' AE on, 'b' AWB on, 'i/o' exposure, 'k/l' ISO, ',/.' focus, WASD crop, locks 1/2, tuning 3..9 0 [ ] + -

Outputs (in session folder):
    - rgb/*.png           : ISP RGB frames (timestamps in filename)
    - imu.csv             : timestamp, gyro (rad/s), linear accel (m/s^2)
    - frames.csv          : per-frame metadata (timestamp, type, seq, exposure_us, iso, path)
"""

import os
import csv
from datetime import datetime

import depthai as dai
import cv2
from itertools import cycle

from rsaai.data_collection.camera_intrinsics import get_oak_camera_intrinsics

STEP_SIZE = 8
EXP_STEP = 500
ISO_STEP = 50
LENS_STEP = 3
WB_STEP = 200

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

pipeline = dai.Pipeline()
imu = pipeline.create(dai.node.IMU)

# IMU: enable calibrated gyro and linear acceleration (fallback to calibrated accelerometer)
try:
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, 400)
except Exception:
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 400)
try:
    # Linear acceleration (gravity removed) when supported
    imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, 480)
except Exception:
    # Fallback to calibrated accelerometer
    try:
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER, 480)
    except Exception:
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 480)
# it's recommended to set both setBatchReportThreshold and setMaxBatchReports to 20 when integrating in a pipeline with a lot of input/output connections
# above this threshold packets will be sent in batch of X, if the host is not blocked and USB bandwidth is available
imu.setBatchReportThreshold(1)
# maximum number of IMU packets in a batch, if it's reached device will block sending until host can receive it
# if lower or equal to batchReportThreshold then the sending is always blocking on device
# useful to reduce device's CPU load  and number of lost packets, if CPU load is high on device side due to multiple nodes
imu.setMaxBatchReports(10)

# -------- Camera node --------
camRgb = pipeline.create(dai.node.Camera)
camRgb.build(dai.CameraBoardSocket.CAM_A)


# -------- Create I/O queues (v3 style) --------
ispOut = camRgb.requestOutput((1280, 720), fps=30)
controlIn = camRgb.inputControl

# Create queues
ispQueue = ispOut.createOutputQueue()

imuQueue = imu.out.createOutputQueue(maxSize=100, blocking=False)

controlQueue = controlIn.createInputQueue()

# (Removed) Still capture pipeline link

# -------- Start pipeline (no dai.Device) --------
pipeline.start()


sendCamConfig = True

# Default manual settings
lensPos = 150
expTime = 1 
sensIso = 800
wbManual = 4000
ae_comp = 0
ae_lock = False
awb_lock = False
saturation = 0
contrast = 0
brightness = 0
sharpness = 0
luma_denoise = 0
chroma_denoise = 0
control = 'none'
show = False

awb_mode = cycle([v for k, v in vars(dai.CameraControl.AutoWhiteBalanceMode).items() if k.isupper()])
anti_banding_mode = cycle([v for k, v in vars(dai.CameraControl.AntiBandingMode).items() if k.isupper()])
effect_mode = cycle([v for k, v in vars(dai.CameraControl.EffectMode).items() if k.isupper()])

# -------- Recording state --------
recording = False
session_dir = None
rgb_dir = None
depth_dir = None
imu_csv_path = None
frames_csv_path = None
imu_csv_file = None
imu_csv_writer = None
frames_csv_file = None
frames_csv_writer = None

def ensure_session():
    global session_dir, rgb_dir, depth_dir, imu_csv_path, frames_csv_path
    global imu_csv_file, imu_csv_writer, frames_csv_file, frames_csv_writer
    if session_dir is None:
        session_dir = os.path.join(
            os.getcwd(),
            f"recordings/session-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        rgb_dir = os.path.join(session_dir, 'rgb')
        os.makedirs(rgb_dir, exist_ok=True)
        imu_csv_path = os.path.join(session_dir, 'imu.csv')
        frames_csv_path = os.path.join(session_dir, 'frames.csv')
        # Open CSVs and write headers
        imu_csv_file = open(imu_csv_path, 'w', newline='')
        imu_csv_writer = csv.writer(imu_csv_file)
        imu_csv_writer.writerow([
            'timestamp_ns', 'gyro_x_rad_s', 'gyro_y_rad_s', 'gyro_z_rad_s',
            'linacc_x_m_s2', 'linacc_y_m_s2', 'linacc_z_m_s2'
        ])
        frames_csv_file = open(frames_csv_path, 'w', newline='')
        frames_csv_writer = csv.writer(frames_csv_file)
        frames_csv_writer.writerow([
            'timestamp_ns', 'type', 'seq', 'exposure_us', 'iso', 'path'
        ])
        print(f"Session folder: {session_dir}")
    return session_dir

def close_session_files():
    global imu_csv_file, frames_csv_file
    try:
        if imu_csv_file is not None:
            imu_csv_file.flush()
            imu_csv_file.close()
    except Exception:
        pass
    try:
        if frames_csv_file is not None:
            frames_csv_file.flush()
            frames_csv_file.close()
    except Exception:
        pass

# =====================================================
#                   Main Loop
# =====================================================
while pipeline.isRunning():

    # ISP frames
    for ispFrame in ispQueue.getAll():
        if show:
            txt = f"[{ispFrame.getSequenceNum()}] "
            txt += f"Exposure: {ispFrame.getExposureTime().total_seconds()*1000:.3f} ms, "
            txt += f"ISO: {ispFrame.getSensitivity()}, "
            txt += f"Lens: {ispFrame.getLensPosition()}, "
            txt += f"Temp: {ispFrame.getColorTemperature()} K"
            print(txt)
        cv2_frame = ispFrame.getCvFrame()
        cv2.imshow("isp", cv2_frame)
        if recording and session_dir is not None:
            ts = ispFrame.getTimestamp()
            ts_ns = int(ts.total_seconds() * 1e9)
            seq = ispFrame.getSequenceNum()
            exposure_us = int(ispFrame.getExposureTime().total_seconds() * 1e6)
            iso = int(ispFrame.getSensitivity())
            rgb_path = os.path.join(rgb_dir, f"rgb_{ts_ns}_{seq}.png")
            cv2.imwrite(rgb_path, cv2_frame)
            if frames_csv_writer is not None:
                frames_csv_writer.writerow([ts_ns, 'rgb', seq, exposure_us, iso, os.path.relpath(rgb_path, session_dir)])
        
        
    # ------------ IMU capture (poll) ------------
    if not recording or session_dir is None or imu_csv_writer is None:
            pass 
    else:
        for imuData in imuQueue.tryGetAll():
            # Each IMUData can contain multiple packets
            packets = imuData.packets if hasattr(imuData, 'packets') else []
            for pkt in packets:
                # Initialize with NaNs for missing fields
                ts = None
                gx = gy = gz = ''
                ax = ay = az = ''
                # Gyro calibrated
                if hasattr(pkt, 'gyroscope') and pkt.gyroscope is not None:
                    g = pkt.gyroscope
                    ts = g.getTimestamp()
                    gx, gy, gz = g.x, g.y, g.z
                # Linear acceleration preferred
                if hasattr(pkt, 'linearAcceleration') and pkt.linearAcceleration is not None:
                    la = pkt.linearAcceleration
                    ts = la.getTimestamp() if ts is None else ts
                    ax, ay, az = la.x, la.y, la.z
                # Fallback to accelerometer if needed
                if (ax == '' or ay == '' or az == '') and hasattr(pkt, 'acceleroMeter') and pkt.acceleroMeter is not None:
                    a = pkt.acceleroMeter
                    ts = a.getTimestamp() if ts is None else ts
                    ax, ay, az = a.x, a.y, a.z
                if ts is None:
                    continue
                ts_ns = int(ts.total_seconds() * 1e9)
                imu_csv_writer.writerow([ts_ns, gx, gy, gz, ax, ay, az])

    key = cv2.waitKey(1)
    if key == ord('q'):
        if recording:
            print("Stopping recording and exiting...")
        break

    # ------------ Recording start ------------
    elif key == ord('r'):
        if not recording:
            recording = True
            ensure_session()
            print("Recording started. Outputs will be saved to:", session_dir)
        else:
            print("Already recording.")

    # ------------ Recording stop -------------
    elif key == ord('s'):
        if recording:
            recording = False
            print("Recording stopped. Files saved in:", session_dir)
        else:
            print("Not currently recording.")

    # ------------ Show toggle ------------
    elif key == ord('/'):
        show = not show
        if not show:
            print("Camera settings print OFF")

    # ------------ Capture still ------------
    elif key == ord('c'):
        ctrl = dai.CameraControl()
        ctrl.setCaptureStill(True)
        controlQueue.send(ctrl)

    # ------------ AF trigger ------------
    elif key == ord('t'):
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.AUTO)
        ctrl.setAutoFocusTrigger()
        controlQueue.send(ctrl)

    # ------------ AF continuous ------------
    elif key == ord('f'):
        ctrl = dai.CameraControl()
        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
        controlQueue.send(ctrl)

    # ------------ Auto exposure ------------
    elif key == ord('e'):
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureEnable()
        controlQueue.send(ctrl)

    # ------------ Auto white balance ------------
    elif key == ord('b'):
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        controlQueue.send(ctrl)

    # ------------ Manual focus ------------
    elif key in [ord(','), ord('.')]:
        if key == ord(','): lensPos -= LENS_STEP
        if key == ord('.'): lensPos += LENS_STEP
        lensPos = clamp(lensPos, 0, 255)
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(lensPos)
        controlQueue.send(ctrl)

    # ------------ Manual exposure / ISO ------------
    elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
        if key == ord('i'): expTime -= EXP_STEP
        if key == ord('o'): expTime += EXP_STEP
        if key == ord('k'): sensIso -= ISO_STEP
        if key == ord('l'): sensIso += ISO_STEP
        expTime = clamp(expTime, 1, 33000)
        sensIso = clamp(sensIso, 100, 1600)
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(expTime, sensIso)
        controlQueue.send(ctrl)

    # ------------ Manual WB ------------
    elif key in [ord('n'), ord('m')]:
        if key == ord('n'): wbManual -= WB_STEP
        if key == ord('m'): wbManual += WB_STEP
        wbManual = clamp(wbManual, 1000, 12000)
        ctrl = dai.CameraControl()
        ctrl.setManualWhiteBalance(wbManual)
        controlQueue.send(ctrl)

    # ------------ Crop WASD ------------
    elif key in [ord('w'), ord('a'), ord('s'), ord('d')]:
        if key == ord('a'):
            cropX -= (maxCropX / camRgb.getResolutionWidth()) * STEP_SIZE
        elif key == ord('d'):
            cropX += (maxCropX / camRgb.getResolutionWidth()) * STEP_SIZE
        elif key == ord('w'):
            cropY -= (maxCropY / camRgb.getResolutionHeight()) * STEP_SIZE
        elif key == ord('s'):
            cropY += (maxCropY / camRgb.getResolutionHeight()) * STEP_SIZE
        cropX = clamp(cropX, 0, maxCropX)
        cropY = clamp(cropY, 0, maxCropY)
        sendCamConfig = True

    # ------------ AE/AWB locks ------------
    elif key == ord('1'):
        awb_lock = not awb_lock
        ctrl = dai.CameraControl()
        ctrl.setAutoWhiteBalanceLock(awb_lock)
        controlQueue.send(ctrl)

    elif key == ord('2'):
        ae_lock = not ae_lock
        ctrl = dai.CameraControl()
        ctrl.setAutoExposureLock(ae_lock)
        controlQueue.send(ctrl)

    # ------------ Select control mode (3..9 0 [ ]) ------------
    elif key >= 0 and chr(key) in "34567890[]":
        if key == ord('3'): control = 'awb_mode'
        elif key == ord('4'): control = 'ae_comp'
        elif key == ord('5'): control = 'anti_banding_mode'
        elif key == ord('6'): control = 'effect_mode'
        elif key == ord('7'): control = 'brightness'
        elif key == ord('8'): control = 'contrast'
        elif key == ord('9'): control = 'saturation'
        elif key == ord('0'): control = 'sharpness'
        elif key == ord('['): control = 'luma_denoise'
        elif key == ord(']'): control = 'chroma_denoise'
        print("Selected control:", control)

    # ------------ Adjust selected control ------------
    elif key in [ord('-'), ord('_'), ord('+'), ord('=')]:
        if control == 'none':
            print("Select a control (3..9 0 [ ]) first")
            continue

        change = -1 if key in [ord('-'), ord('_')] else 1
        ctrl = dai.CameraControl()

        if control == 'ae_comp':
            ae_comp = clamp(ae_comp + change, -9, 9)
            ctrl.setAutoExposureCompensation(ae_comp)

        elif control == 'anti_banding_mode':
            ctrl.setAntiBandingMode(next(anti_banding_mode))

        elif control == 'awb_mode':
            ctrl.setAutoWhiteBalanceMode(next(awb_mode))

        elif control == 'effect_mode':
            ctrl.setEffectMode(next(effect_mode))

        elif control == 'brightness':
            brightness = clamp(brightness + change, -10, 10)
            ctrl.setBrightness(brightness)

        elif control == 'contrast':
            contrast = clamp(contrast + change, -10, 10)
            ctrl.setContrast(contrast)

        elif control == 'saturation':
            saturation = clamp(saturation + change, -10, 10)
            ctrl.setSaturation(saturation)

        elif control == 'sharpness':
            sharpness = clamp(sharpness + change, 0, 4)
            ctrl.setSharpness(sharpness)

        elif control == 'luma_denoise':
            luma_denoise = clamp(luma_denoise + change, 0, 4)
            ctrl.setLumaDenoise(luma_denoise)

        elif control == 'chroma_denoise':
            chroma_denoise = clamp(chroma_denoise + change, 0, 4)
            ctrl.setChromaDenoise(chroma_denoise)

        controlQueue.send(ctrl)

# -------- Cleanup --------
close_session_files()
