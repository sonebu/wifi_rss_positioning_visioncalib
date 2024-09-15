import argparse, os, sys
import time

if __name__ == "__main__":
    # construct the argument parser and parse the arguments

    start_time=time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--experiment-folderpath", required=True, help="path to the experiment we want to log location data for")
    args = vars(ap.parse_args())

    # check if the submitted experiment is a valid one
    expcorrect = False
    if(os.path.isdir(args["experiment_folderpath"]) is False):
        print("You must provide an existing experiment name")
    else:
        if(os.path.exists(os.path.join(args["experiment_folderpath"], "devstring.txt")) is False):
            print("The experiment does not contain a devstring.txt file (it should)")
        else:
            with open(os.path.join(args["experiment_folderpath"], "devstring.txt"), "r") as f:
                lines = [line.rstrip() for line in f]
            if len(lines) != 3:
                print("The devstring does not have 3 lines like it should, the experiment might not be an appropriate one")
            else:
                expcorrect = True

    if expcorrect:
        from ultralytics import YOLO
        import numpy as np
        import cv2, datetime, time, cvzone
        from collections import deque
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure
        from PyQt5 import QtWidgets, QtGui
        from PyQt5.QtGui import QPixmap, QFont
        from PyQt5.QtWidgets import QLabel, QWidget, QFrame, QPushButton, QLineEdit, QVBoxLayout, QMessageBox
        from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QPoint, QMetaObject, QCoreApplication

        from calibration_gui_detectaruco import aruco_detect_draw

        class LocationMap(FigureCanvas):
            """A QWidget embedding a Matplotlib plot."""
            
            def __init__(self, parent=None):
                fig = Figure()
                self.axes = fig.add_subplot(111)
                self.right_axes = self.axes.twinx()  # Create right y-axis
                super(LocationMap, self).__init__(fig)
                self.setParent(parent)
                self.axes.grid(True)
   

            def update_plot(self, loc_data, input_val):
                """Update the plot with the last 5 seconds of location data."""
                self.axes.clear()
                self.right_axes.clear()
                           
                
                
                self.axes.tick_params(axis='y', colors='blue')  # Set tick labels and ticks color to blue
                self.axes.spines['left'].set_color('blue')  # Set the spine color to blue

                # Set the color of the right y-axis to red
                
                self.right_axes.tick_params(axis='y', colors='red')  # Set tick labels and ticks color to red
                self.right_axes.spines['right'].set_color('red') 
				
                if loc_data:
                    times = [t-start_time for t, _, _ in loc_data]  # Normalize time to 0 at the beginning
                    x_coords = [x for _, x, _ in loc_data]
                    y_coords = [y for _, _, y in loc_data]
                    
                    x_coords_np = np.array(x_coords)
                    y_coords_np = np.array(y_coords)
                    times_np=np.array(times)
                    valid_x = ~np.isnan(x_coords_np)
                    valid_y = ~np.isnan(y_coords_np)

                    # Replace invalid (NaN) values with NaN in coordinates, but keep times intact
                    x_coords_np = np.where(valid_x, x_coords_np, np.nan)
                    y_coords_np = np.where(valid_y, y_coords_np, np.nan)
                   
                    if np.any(valid_x):
                        x_min, x_max = np.nanmin(x_coords_np), np.nanmax(x_coords_np)
                    else:
                        x_min, x_max = -input_val, input_val  # Default limits in case of no valid x data

                    if np.any(valid_y):
                        y_min, y_max = np.nanmin(y_coords_np), np.nanmax(y_coords_np)
                    else:
                        y_min, y_max = -input_val, input_val # Default limits in case of no valid y data

                    x_plot, = self.axes.plot(times_np, x_coords_np, 'b-', label="X")
                    # Plot x_loc on left y-axis, keeping time continuous
                    self.axes.plot(times_np, x_coords_np, 'b-')
                    self.axes.set_xlim(max(0, times_np[0]), max(5, times_np[-1]))  # Adjust x-axis to show last 5 seconds
                    self.axes.set_ylim(x_min-input_val, x_max+input_val)  # Set y-axis limits based on valid x data

                    # Plot y_loc on right y-axis, keeping time continuous
                    y_plot, = self.right_axes.plot(times_np, y_coords_np, 'r-', label="Y")
       
                    self.right_axes.plot(times_np, y_coords_np, 'r-')
                    self.right_axes.set_ylim(y_min-input_val, y_max+input_val)  # Set y-axis limits based on valid y data
                    self.axes.legend(handles=[x_plot, y_plot], loc='upper right')


                self.draw()

        class Ui_MainWindow(object):
            def setupUi(self, MainWindow):
                if not MainWindow.objectName():
                    MainWindow.setObjectName(u"MainWindow")
                MainWindow.resize(1800, 800)
                self.centralwidget = QWidget(MainWindow)
                self.centralwidget.setObjectName(u"centralwidget")
                font = QFont()
                font.setPointSize(25)
                self.label_cordx = QLabel(self.centralwidget)
                self.label_cordx.setObjectName(u"label")
                self.label_cordx.setGeometry(QRect(1300, 10, 350, 160))
                self.label_cordx.setFont(font)
                self.label_cordx.setAutoFillBackground(False)
                self.label_cordx.setFrameShape(QFrame.NoFrame)
                self.label_cordx.setWordWrap(True)
                self.label_cordy = QLabel(self.centralwidget)
                self.label_cordy.setObjectName(u"label2")
                self.label_cordy.setGeometry(QRect(1300, 200, 350, 160))
                self.label_cordy.setFont(font)
                self.label_cordy.setAutoFillBackground(False)
                self.label_cordy.setFrameShape(QFrame.NoFrame)
                self.label_cordy.setWordWrap(True)
                self.location_map = LocationMap(self.centralwidget)
                self.location_map.setGeometry(QRect(1325, 400, 420, 400))

                self.input_box = QLineEdit(self.centralwidget)
                self.input_box.setGeometry(QRect(1325, 350, 100, 40))
                self.input_box.setPlaceholderText("Enter a number")
                self.input_box.textChanged.connect(self.validate_input)


                self.validate_button = QPushButton("Validate", self.centralwidget)
                self.validate_button.setGeometry(QRect(1425, 350, 100, 40))
                self.validate_button.clicked.connect(self.confirm_input)
                
                # Add button for starting camera
                self.start_camera_button = QPushButton("Start Camera", self.centralwidget)
                self.start_camera_button.setGeometry(QRect(1550, 350, 200, 40))
                self.start_camera_button.setEnabled(False) 
                

                
                MainWindow.setCentralWidget(self.centralwidget)
                self.retranslateUi(MainWindow)
                QMetaObject.connectSlotsByName(MainWindow)

            def retranslateUi(self, MainWindow):
                MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
                self.label_cordx.setText(QCoreApplication.translate("MainWindow", u"x coord.", None))
                self.label_cordy.setText(QCoreApplication.translate("MainWindow", u"y coord.", None))

            def validate_input(self):
                text = self.input_box.text()
                try:
                    # Check if input can be converted to float
                    float(text)
                    self.input_box.setStyleSheet("border: 1px solid green;")  # Valid input
                      # Enable start button
                except ValueError:
                    self.input_box.setStyleSheet("border: 1px solid red;")  # Invalid input
                    self.start_camera_button.setEnabled(False)  # Keep start button disabled

            def confirm_input(self):
                """Confirm if the user is sure about the input and enable the start button if valid."""
                text = self.input_box.text()
                try:
                    value = float(text)  # Try to convert input to float
                    confirmation = QMessageBox.question(
                        self.centralwidget, 
                        "Confirm Input", 
                        f"Are you sure about the value: {value}?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if confirmation == QMessageBox.Yes:
                        self.input_val = value
                        self.start_camera_button.setEnabled(True)  # Enable start button
                        self.validate_button.setEnabled(False)
                        self.input_box.setEnabled(False)
                        self.start_camera_button.setEnabled(True)
                    else:
                        self.start_camera_button.setEnabled(False)  # Keep it disabled
                except ValueError:
                    QMessageBox.warning(self.centralwidget, "Invalid Input", "Please enter a valid number.")




        class VideoThread(QThread):
            change_pixmap_signal = pyqtSignal(np.ndarray)
            update_loc_signal = pyqtSignal(list)

            def __init__(self, cap_obj, label_cordx, label_cordy, location_map, input_val):
                super().__init__()
                self._run_flag = True
                self.cap_obj = cap_obj
                self.cv_img_memory = None
                self.model = YOLO("yolo-Weights/yolov8n.pt")
                self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                                   "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                   "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                                   "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                                   "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                                   "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                                   "teddy bear", "hair drier", "toothbrush"]
                self.homography_matrix = np.load(os.path.join(args["experiment_folderpath"], "homography_matrix.npy"))
                self.loc_xy_file = open(os.path.join(args["experiment_folderpath"], "loc_xy.txt"), "w")
                self.label_cordx = label_cordx
                self.label_cordy = label_cordy
                self.location_map = location_map
                self.input_val = input_val
                self.loc_data_buffer = deque(maxlen=500)

            def run(self):
                while self._run_flag:
                    ret, cv_img = self.cap_obj.read()
                    if ret:
                        # run YOLO
                        yolo_predictions = self.model(cv_img, stream=True, verbose=False)
                        yolo_detections = np.empty((0, 5))

                        # run aruco
                        centers, corners, ids, cv_img = aruco_detect_draw(cv_img, verbose=False, draw=True)

                        for r in yolo_predictions:
                            for box in r.boxes:
                                classname = int(box.cls[0])  # get detected class name
                                if self.classNames[classname] == "person":
                                    x1, y1, x2, y2 = box.xyxy[0]
                                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                                    is_detected_person_arucotagged = False
                                    for ii, center in enumerate(centers):
                                        is_arucotag_id_correct = True if ids[ii] == 0 else False
                                        is_tag_within_box = True if ((center[0] >= x1) and (center[0] <= x2)) and (
                                                    (center[1] >= y1) and (center[1] <= y2)) else False
                                        if is_arucotag_id_correct and is_tag_within_box:
                                            is_detected_person_arucotagged = True
                                    if is_detected_person_arucotagged:
                                        confidence = np.ceil((box.cpu().conf[0] * 100)) / 100  # cast to .cpu()
                                        current_detection = np.array([[x1, y1, x2, y2, confidence]])
                                        yolo_detections = np.vstack((yolo_detections, current_detection))
                                        current_loc = [(x1 + x2) // 2, y2]  # get midpoint at feet
                                        current_cord = cv2.perspectiveTransform(np.array([[current_loc]], dtype=np.float32),
                                                                                self.homography_matrix)
                                        current_cordx = current_cord[0][0][0]
                                        current_cordy = current_cord[0][0][1]
                                        timestamp = time.time()
                                        timestamp_ordered = datetime.datetime.fromtimestamp(timestamp).strftime(
                                            "%H:%M:%S.%f")
                                        w, h = x2 - x1, y2 - y1
                                        cvzone.cornerRect(cv_img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 255, 0))
                                        self.loc_xy_file.write(
                                            timestamp_ordered + "," + str(current_cordx) + "," + str(current_cordy) + "\n")
                                        self.label_cordx.setText(f"loc_X: {np.round(1000 * current_cordx) / 1000}")
                                        self.label_cordy.setText(f"loc_Y: {np.round(1000 * current_cordy) / 1000}")
                                        self.loc_data_buffer.append((timestamp, current_cordx, current_cordy))
                                        self.filter_last_5_seconds()
                                        self.update_loc_display()
                                    else:
                                        w, h = x2 - x1, y2 - y1
                                        cvzone.cornerRect(cv_img, (x1, y1, w, h), l=9, rt=2, colorR=(0, 0, 255))
                                        timestamp = time.time()
                                        current_cordx = np.nan
                                        current_cordy = np.nan
                                        self.loc_data_buffer.append((timestamp, current_cordx, current_cordy))
                                        self.filter_last_5_seconds()
                                        self.update_loc_display()
                                        

                        self.change_pixmap_signal.emit(cv_img)

                # shut down capture system
                self.cap_obj.release()

            def stop(self):
                """Sets run flag to False and waits for thread to finish"""
                self._run_flag = False
                self.loc_xy_file.close()
                self.wait()

            def filter_last_5_seconds(self):
                current_time = time.time()
                # Keep only data points within the last 5 seconds
                self.loc_data_buffer = deque([(t, x, y) for (t, x, y) in self.loc_data_buffer if current_time - t <= 5])

            def update_loc_display(self):
                """Updates the last 5 seconds of loc data in the plot."""
                self.location_map.update_plot(list(self.loc_data_buffer), self.input_val)

        class ClickableFrameQLabel(QLabel):
            clickpos = pyqtSignal(QPoint)

            def mousePressEvent(self, event):
                self.clickpos.emit(event.pos())
                QLabel.mousePressEvent(self, event)

        class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
            def __init__(self, parent=None, cap=None):
                super(MainWindow, self).__init__(parent=parent)

                # Get the layout up
                self.setupUi(self)
                self.frame = ClickableFrameQLabel(self.centralwidget)  # the picture frame is created separately
                self.frame.setGeometry(QRect(10, 10, 1280, 720))
                self.start_camera_button.clicked.connect(self.start_camera)
                self.cap=cap
                self.cv2thread =None
            def start_camera(self):
                if self.cap:
                # to avoid premature exit warnings
                    self.cv2thread = VideoThread(cap, self.label_cordx, self.label_cordy, self.location_map, self.input_val)
                    self.cv2thread.change_pixmap_signal.connect(self.update_image)
                    self.cv2thread.start()
                    self.input_box.setEnabled(False)  # Disable input box after starting the camera
                    self.start_camera_button.setEnabled(False)

                # some buffers
                self.cv_img_buffer = None

            def closeEvent(self, event):
                if self.cv2thread is not None:
                    self.cv2thread.stop()
                event.accept()

            @pyqtSlot(np.ndarray)
            def update_image(self, cv_img):
                qt_img = self.convert_cv_qt(cv_img)
                self.frame.setPixmap(qt_img)

            def convert_cv_qt(self, cv_img):
                self.cv_img_buffer = cv_img.copy()
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(1280, 720, Qt.KeepAspectRatio)
                return QPixmap.fromImage(p)

            def keyPressEvent(self, e):
                if e.key() == Qt.Key_Q:
                    self.close()

        # start webcam
        cap = cv2.VideoCapture(lines[0])
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # this should match calibration_gui.py
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(lines[1]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(lines[2]))
        actual_cam_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cam_ratio = float(min((1280, 720))) / min(actual_cam_res)  # 1280x720 is the interface size

        if (actual_cam_res[0] != int(lines[1])) and (actual_cam_res[1] != int(lines[2])):
            print("The cam could not obtain the desired resolution for some reason, debug this before continuing")
        else:
            # if everything's OK, start GUI
            app = QtWidgets.QApplication(sys.argv)
            w = MainWindow(cap=cap)
            w.show()
            sys.exit(app.exec_())
    else:
        print("exiting")
