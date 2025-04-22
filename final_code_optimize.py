# to install packages for model "req_pack"

# ------ Use => pip install -r req_pack.txt

import cv2
import mediapipe as mp
import numpy as np

class distance_check:

    def __init__(self, mp_pose):
        self.pose = mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp_pose
        self.landmark_index = mp_pose.PoseLandmark.RIGHT_HIP.value
        self.y_positions = []

    def video_capture(self):
        cap = cv2.VideoCapture(0)
        while True:
            re, frame = cap.read()
            if not re:
                break
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.pose.process(frame_rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                landmark = landmarks[self.landmark_index]
                y_pixel = int(landmark.y * h)
                x_pixel = int(landmark.x * w)
                self.y_positions.append(y_pixel)

                line_length = 20
                start_point = (x_pixel - line_length // 2, y_pixel)
                end_point = (x_pixel + line_length // 2, y_pixel)
                cv2.line(frame, start_point, end_point, (255, 0, 0), 3)

                self.mp_drawing.draw_landmarks(frame, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

# ------------ frame will show jumping distance ------------------

                # cv2.putText(frame, f"Y Pos: {self.jump_height_meters}", (30, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Jump Estimation", frame)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def estimate_jump_height(self, y_data, pixel_to_meter_ratio=0.0025):
        if not y_data:
            return 0.0
        y_min = min(y_data)  
        y_max = max(y_data) 
        pixel_displacement = y_max - y_min
        jump_height_meters = pixel_displacement * pixel_to_meter_ratio
        return jump_height_meters

mp_pose = mp.solutions.pose
estimator = distance_check(mp_pose)  
estimator.video_capture()

# Estimate and print jump height
jump_height = estimator.estimate_jump_height(estimator.y_positions)

print(f"Estimated Jump Height: {jump_height:.2f} meters")  # Result will mention below
