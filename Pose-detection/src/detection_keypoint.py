import sys
import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO
import ultralytics
from ultralytics.engine.results import Results

# Define keypoint
class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_SHOULDER:  int = 1
    RIGHT_SHOULDER: int = 2
    LEFT_HIP:       int = 3
    RIGHT_HIP:      int = 4
    LEFT_KNEE:      int = 5
    RIGHT_KNEE:     int = 6
    LEFT_ANKLE:     int = 7
    RIGHT_ANKLE:    int = 8

class DetectKeypoint:
    def __init__(self, yolov8_model='yolov8m-pose'):
        self.yolov8_model = yolov8_model
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    def __load_model(self):
        if not self.yolov8_model.split('-')[-1] == 'pose':
            sys.exit('Model not yolov8 pose')
        self.model = ultralytics.YOLO(model=self.yolov8_model)

        # extract function keypoint
    def extract_keypoint(self, keypoint: np.ndarray) -> list:
        # nose
        nose_x, nose_y = keypoint[self.get_keypoint.NOSE]
        # eye
        left_shoulder_x, left_shoulder_y = keypoint[self.get_keypoint.LEFT_SHOULDER]
        right_shoulder_x, right_shoulder_y = keypoint[self.get_keypoint.RIGHT_SHOULDER]
        # elbow
    
        # hip
        left_hip_x, left_hip_y = keypoint[self.get_keypoint.LEFT_HIP]
        right_hip_x, right_hip_y = keypoint[self.get_keypoint.RIGHT_HIP]
        # knee
        left_knee_x, left_knee_y = keypoint[self.get_keypoint.LEFT_KNEE]
        right_knee_x, right_knee_y = keypoint[self.get_keypoint.RIGHT_KNEE]
        # ankle
        left_ankle_x, left_ankle_y = keypoint[self.get_keypoint.LEFT_ANKLE]
        right_ankle_x, right_ankle_y = keypoint[self.get_keypoint.RIGHT_ANKLE]
        
        return [
            nose_x, nose_y,left_shoulder_x, left_shoulder_y,
            right_shoulder_x, right_shoulder_y, left_hip_x, left_hip_y,
            right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y,        
            left_ankle_x, left_ankle_y,right_ankle_x, right_ankle_y
        ]
    
    def get_xy_keypoint(self, results: Results) -> list:
        result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data
    
    def __call__(self, image: np.array) -> Results:
        results = self.model.predict(image, save=False)[0]
        return results

