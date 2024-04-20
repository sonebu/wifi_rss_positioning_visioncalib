from ultralytics import YOLO
import cv2
import math 
import numpy as np
import json
# start webcam
cap = cv2.VideoCapture("/dev/video2")
cap.set(3, 1920)  # Set width to 1920
cap.set(4, 1080)  # Set height to 1080

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

filename = "/home/kadir/ELEC491/calpnp/points.txt"
with open(filename, 'r') as file:
    lines = file.readlines()
coordinates = []

for line in lines:
    x, y = map(int, line.split())
    coordinates.append([x, y])

cal2dPtLs = np.array(coordinates)
"""

cal2dPtLs = np.array([[627, 779], [736, 204], [831, 155], [1092, 157],
                             [1187, 197], [1320, 773], [1200, 888], [753, 895]])
"""
cal3dPtLs = np.array([[-5, -8], [-5, 8], [-3, 10], [3, 10],
                       [5, 8], [5, -8], [3, -10], [-3, -10]], dtype=np.float32)

homography_matrix, _ = cv2.findHomography(cal2dPtLs, cal3dPtLs)


loc_data=[]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object detailsr
            if classNames[cls]=="person":
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                
                cur_loc = [(x1 + x2) // 2,y2]
                
                
                test_points = np.array([cur_loc], dtype=np.float32)

                transformed_points = cv2.perspectiveTransform(np.array([test_points]), homography_matrix)
                
                loc_data.append(transformed_points[0])
                    
                

    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        for point in loc_data:
            x_loc = point[0][0]
            y_loc = point[0][1]
            print(f"[{x_loc},{y_loc}],")
            #Fileya yazdirilacak (Yazdirmayi denedim ama yazdirmiyor)
        break

cap.release()
cv2.destroyAllWindows()



