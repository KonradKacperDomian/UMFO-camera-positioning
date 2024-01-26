from typing import Any
from imutils.video import VideoStream
import numpy as np
import threading
import imutils
import time
import cv2
from cv2.typing import MatLike
import sys

sys.path.append("/home/raspberrypi/UMFO-camera-positioning/IMU")
from IMU import *

class Marker_Detector(threading.Thread):
    def __init__(self) -> None:
        super(Marker_Detector, self).__init__()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.MAIN_MARKER_ID = "1"
        self.MARKER_WIDTH = 53                       # value specified in mm
        self.MARKER_HEIGHT = 53                      # value specified in mm
        self.markers_position = self.initialize_markers_position()
        self.currnet_distance = [0.0, 0.0, 0.0]
        self.currnet_rotation = [0.0, 0.0, 0.0]
        self.CCD_WIDTH = 1280
        self.CCD_HEIGHT = 720 
        self.FOCAL_LENGTH_X = 1090.79
        self.FOCAL_LENGTH_Y = 1092.72
        self.M2_TL_M1_TL_X = 120                    # Distance from marker 2 top left point to marker 1 top left point in x axis [mm]
        self.M2_TL_M1_TL_Y = 75                     # Distance from marker 2 top left point to marker 1 top left point in y axis [mm]

    def initialize_markers_position(self) -> dict:
        distance_dict = {
            "1" : [(0, 0), (0, 0), (0, 0), (0, 0)],
            "2" : [(0, 0), (0, 0), (0, 0), (0, 0)],
            "3" : [(0, 0), (0, 0), (0, 0), (0, 0)],
            "4" : [(0, 0), (0, 0), (0, 0), (0, 0)],
            "5" : [(0, 0), (0, 0), (0, 0), (0, 0)]
        }
        return distance_dict

    def print_data(self):
        x, y, z = self.currnet_distance
        #roll, pitch, yaw = imu.current_rotation
        print("Position: {:<5}{:>6.2f}   {:<5}{:>6.2f}   {:<5}{:>6.2f}  cm".format("X:", round(x)/10, "Y:", round(y)/10, "Z:", round(z)/10))
        #print("{:<6} {:>7.2f}   {:<6} {:>7.2f}   {:<6} {:>7.2f} degree".format("Roll:", roll, "Pitch:", pitch, "Yaw:", yaw))

    def initialize_video_stream(self) -> VideoStream:
        video_stream = VideoStream(src = 0).start()
        time.sleep(2.0)
        return video_stream
    
    def get_marker_position(self, key : str) -> list:
        if not isinstance(key, str):
            key = str(key)
        return self.markers_position[key]
    
    def convert_marker_coordinate(self, marker_corners : np.ndarray):
        corners = marker_corners.reshape((4, 2))
        top_left, top_right, bottom_right, bottom_left = corners
        top_left = (int(top_left[0]), int(top_left[1]))
        top_right = (int(top_right[0]), int(top_right[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
        return top_left, top_right, bottom_right, bottom_left

    def update_marker_position(self, key : str, top_left : tuple[int, int], top_right : tuple[int, int], bottom_right : tuple[int, int], bottom_left : tuple[int, int]) -> None:
        if not isinstance(key, str):
            key = str(key)
        new_marker_position = [top_left, top_right, bottom_right, bottom_left]
        self.markers_position[key] = new_marker_position

    def get_marker_px_width(self, key : str) -> int:
        if not isinstance(key, str):
            key = str(key)
        top_left, top_right, _, _ = self.get_marker_position(key)
        marker_px_width = abs(top_right[0] - top_left[0])
        return marker_px_width

    def get_marker_px_height(self, key: str) -> int:
        if not isinstance(key, str):
            key = str(key)
        top_left, _, _, bottom_left =  self.get_marker_position(key)
        marker_px_height = abs(top_left[1] - bottom_left[1])
        return marker_px_height

    def get_marker_center_point(self, key: str) -> tuple:
        if not isinstance(key, str):
            key = str(key)
        top_left, _, bottom_right, _ = self.get_marker_position(key)
        cX = int((top_left[0] + bottom_right[0]) / 2.0)
        cY = int((top_left[1] + bottom_right[1]) / 2.0)
        return (cX, cY)
    
    def get_distance_between(self, id_1 : str, id_2 : str, axis : int) -> int:
        if not isinstance(id_1, str):
            id_1 = str(id_1)
        if not isinstance(id_2, str):
            id_2 = str(id_2)
        if axis > 1:
            axis = 1
        elif axis < 0:
            axis = 0
        top_left_1, _, _, _ = self.get_marker_position(id_1)
        top_left_2, _, _, _ = self.get_marker_position(id_2)
        distance = top_left_1[axis] - top_left_2[axis]
        return distance

    def draw_bounding_box(self, frame : MatLike, key : str) -> None:
        if not isinstance(key, str):
            key = str(key)
        top_left, top_right, bottom_right, bottom_left = self.get_marker_position(key)
        marker_center_point = self.get_marker_center_point(key)
        cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
        cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
        cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
        cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)
        cv2.circle(frame, marker_center_point, 4, (0, 0, 255), -1)
        cv2.putText(frame, key, (top_left[0], top_left[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def get_camera_marker_X_distance(self) -> float:
        main_marker_px_width = self.get_marker_px_width(self.MAIN_MARKER_ID)
        if main_marker_px_width == 0:
            return 0.0
        X_distance = self.MARKER_WIDTH * self.FOCAL_LENGTH_X / main_marker_px_width
        return X_distance + 98.19

    def get_camera_marker_Y_distance(self) -> float:
        camera_center_x = self.CCD_WIDTH // 2
        marker_center_x, _ = self.get_marker_center_point(self.MAIN_MARKER_ID)
        M2_M1_px_distance = self.get_distance_between("2", self.MAIN_MARKER_ID, 0)
        if M2_M1_px_distance == 0:
            return 0.0
        scale_coef = self.M2_TL_M1_TL_X / M2_M1_px_distance
        Y_distance = ((marker_center_x - camera_center_x) * self.get_camera_marker_X_distance()) / self.FOCAL_LENGTH_X
        return Y_distance

    def get_camera_marker_Z_distance(self) -> float:
        camera_center_y = self.CCD_HEIGHT // 2
        _, marker_center_y = self.get_marker_center_point(self.MAIN_MARKER_ID)
        M2_M1_px_distance = self.get_distance_between("2", self.MAIN_MARKER_ID, 1)
        if M2_M1_px_distance == 0:
            return 0.0
        scale_coef = self.M2_TL_M1_TL_Y / M2_M1_px_distance
        Z_distance = ((marker_center_y - camera_center_y) * self.get_camera_marker_X_distance()) / self.FOCAL_LENGTH_Y
        return Z_distance

    def set_current_distance(self) -> None:
        x = self.get_camera_marker_X_distance()
        y = self.get_camera_marker_Y_distance()
        z = self.get_camera_marker_Z_distance()
        new_distance = [x, y, z]
        self.currnet_distance = new_distance

    def set_current_rotation(self):
        pass

    def draw_central_point(self, frame : MatLike) -> None:
        cx = self.CCD_WIDTH // 2
        cy = self.CCD_HEIGHT // 2
        cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)

    def run(self) -> None:
        video_stream = self.initialize_video_stream()
        while True:
            frame = video_stream.read()
            frame = imutils.resize(frame, width = self.CCD_WIDTH, height = self.CCD_HEIGHT)
            self.draw_central_point(frame)
            markers_corners, ids, rejected = self.detector.detectMarkers(frame)
            if len(markers_corners) > 0:
                ids = ids.flatten()
                for (marker_corners, marker_id) in zip(markers_corners, ids):
                    top_left, top_right, bottom_right, bottom_left = self.convert_marker_coordinate(marker_corners)
                    self.update_marker_position(str(marker_id), top_left, top_right, bottom_right, bottom_left)
                    self.draw_bounding_box(frame, marker_id)    
                self.set_current_distance()
                
            #cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()
        video_stream.stop()

if __name__ == "__main__":
    detector = Marker_Detector()
    imu = IMU_thread(terminal_print = False)
    detector.start()
    imu.start()

    while True:
        time.sleep(1)
        imu.print_data(imu.current_movement, imu.rotation, imu.current_position)
        detector.print_data()
        