from ultralytics import YOLO
import cv2
import math 
import numpy as np
import json
from PIL import ImageTk
import datetime
import time
from sort.sort import Sort
import cvzone
# start webcam
cap = cv2.VideoCapture("/dev/video2")
cap.set(3, 1920)  # Set width to 1920
cap.set(4, 1080)  # Set height to 1080

# model
model = YOLO("yolo-Weights/yolov8n.pt")
tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)   # create instance of the SORT tracker

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


cal2dPtLs = np.array([[1490, 685], [1440, 666], [1160, 638], [1095 ,646],
                             [862, 847], [890, 897], [1351, 975], [1451, 942]])
"""
cal3dPtLs = np.array([[6,-2], [7,-1], [7, 4], [6, 5],
                       [-3, 5], [-4, 4], [-4, -1], [-3, -2]], dtype=np.float32)

homography_matrix, _ = cv2.findHomography(cal2dPtLs, cal3dPtLs)


loc_data=[]
id_arr=[]
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    detections = np.empty((0, 5))
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
           

            # class name
            cls = int(box.cls[0])
           

            # object detailsr
            if classNames[cls]=="person":
                #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

               
                cvzone.putTextRect(img, f'{classNames[cls]}', (x2,y2), scale=1,thickness=2,colorR=(0,0,0))
                currentArray = np.array([[x1, y1, x2, y2, confidence]])
                detections = np.vstack((detections, currentArray))
                
                cur_loc = [(x1 + x2) // 2,y2]
                
                
                test_points = np.array([cur_loc], dtype=np.float32)

                transformed_points = cv2.perspectiveTransform(np.array([test_points]), homography_matrix)
                
                loc_data.append(transformed_points[0])
                timestamp=time.time()
                x= datetime.datetime.fromtimestamp(timestamp)
                print(x.strftime("%H:%M:%S.%f"),transformed_points[0], end=" ")
                transformed_x, transformed_y = transformed_points[0][0]
                cvzone.putTextRect(img, f'({transformed_x:.2f}, {transformed_y:.2f})', (x2, y2 + 30), scale=1, thickness=2, colorR=(0, 0, 0))

    result_tracker = tracker.update(detections)

    for res in result_tracker:
        x1, y1, x2, y2, id = res    
        x1, y1, x2, y2,id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1

        cvzone.putTextRect(img, f'ID: {id}', (x1, y1), scale=1,thickness=1,colorR=(0,0,0))
        cvzone.cornerRect(img, (x1, y1, w, h),l=9,rt=1,colorR=(0,0,0))

       
    print(id)


    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        print(len(loc_data))
        print(len(id_arr))
              
        for point in loc_data:
            x_loc = point[0][0]
            y_loc = point[0][1]
            
            
            #print(f"[{id} --> {x_loc},{y_loc}],")
        break

cap.release()
cv2.destroyAllWindows()



