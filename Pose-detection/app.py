import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification

# Initialize the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Initialize keypoint detection and classification models
detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification('./models/pose_classification.pth')

# Function for pose classification on a single frame
def pose_classification(image_cv):
    results = detection_keypoint(image_cv)
    results_keypoint = detection_keypoint.get_xy_keypoint(results)
    image_draw = results.plot(boxes=False)
    x_min, y_min, x_max, y_max = results.boxes.xyxy[0].numpy()

    return image_draw

# Show webcam feed and process the video
def webcam_feed():
    cap = cv2.VideoCapture("/dev/video2")
    cap.set(3, 1920)  # Set width to 1920
    cap.set(4, 1080)  # Set height to 1080

    while True:
        success, img = cap.read()
        if not success:
            break

        image_draw = pose_classification(img)

        cv2.imshow('Webcam Feed', image_draw)

        # Add a break condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the webcam feed function
webcam_feed()

