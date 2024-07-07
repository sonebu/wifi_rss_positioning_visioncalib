import argparse, os, sys

if __name__ == "__main__":
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--experiment-folderpath", required=True, help="path to the experiment we want to log location data for")
	args = vars(ap.parse_args())

	# check if the submitted experiment is a valid one
	expcorrect = False
	if(os.path.isdir(args["experiment_folderpath"]) is False):
		print("You must provide an existing experiment name")
	else:
		if(os.path.exists(os.path.join(args["experiment_folderpath"],"devstring.txt")) is False):
			print("The experiment does not contain a devstring.txt file (it should)")
		else:
			with open(os.path.join(args["experiment_folderpath"],"devstring.txt"),"r") as f:
				lines = [line.rstrip() for line in f]
			if(len(lines) != 3):
				print("The devstring does not have 3 lines like it should, the experiment might not be an appropriate one")
			else:
				expcorrect = True
	if(expcorrect):
		from ultralytics import YOLO
		import numpy as np
		import cv2, datetime, time, cvzone

		from PyQt5 import QtWidgets, QtGui
		from PyQt5.QtGui import QPixmap, QFont
		from PyQt5.QtWidgets import QLabel, QWidget, QFrame
		from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QPoint, QMetaObject, QCoreApplication

		from calibration_gui_detectaruco import aruco_detect_draw

		## disabling SORT functionality for now, we'll use aruco tags to identify the person
		#from datacollection_loc_sort import Sort 

		# GUI
		class Ui_MainWindow(object):
			def setupUi(self, MainWindow):
				if not MainWindow.objectName():
					MainWindow.setObjectName(u"MainWindow")
				MainWindow.resize(1700, 740)
				self.centralwidget = QWidget(MainWindow)
				self.centralwidget.setObjectName(u"centralwidget")
				font = QFont()
				font.setPointSize(50)
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
				MainWindow.setCentralWidget(self.centralwidget)
				self.retranslateUi(MainWindow)
				QMetaObject.connectSlotsByName(MainWindow)

			def retranslateUi(self, MainWindow):
				MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
				self.label_cordx.setText(QCoreApplication.translate("MainWindow", u"x coord.", None))
				self.label_cordy.setText(QCoreApplication.translate("MainWindow", u"y coord.", None))

		# adapted from: https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
		class VideoThread(QThread):
			change_pixmap_signal = pyqtSignal(np.ndarray)

			def __init__(self, cap_obj, label_cordx, label_cordy):
				super().__init__()
				self._run_flag = True
				self.cap_obj = cap_obj;
				self.cv_img_memory	= None;
				self.model   = YOLO("yolo-Weights/yolov8n.pt")
				
				## disabling SORT functionality for now, we'll use aruco tags to identify the person
				#self.tracker = Sort(max_age=10, min_hits=1, iou_threshold=0.3)   # create instance of the SORT tracker
				
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
				self.homography_matrix = np.load(os.path.join(args["experiment_folderpath"],"homography_matrix.npy"))
				self.loc_xy_file = open(os.path.join(args["experiment_folderpath"],"loc_xy.txt"),"w");
				self.label_cordx = label_cordx
				self.label_cordy = label_cordy

			def run(self):
				while self._run_flag:
					ret, cv_img = self.cap_obj.read()
					if ret:
						### buffer image (no need, skip it for now)
						# self.cv_img_memory = cv_img.copy()

						# run YOLO
						yolo_predictions = self.model(cv_img, stream=True, verbose=False)
						yolo_detections  = np.empty((0, 5))
						# run aruco
						centers, corners, ids, cv_img = aruco_detect_draw(cv_img, verbose=False, draw=True)
						for r in yolo_predictions:
							for box in r.boxes:
								classname = int(box.cls[0]) # get detected class name
								if self.classNames[classname]=="person":
									x1, y1, x2, y2     = box.xyxy[0]
									x1, y1, x2, y2     = int(x1), int(y1), int(x2), int(y2) # convert to int values
									is_detected_person_arucotagged = False
									for ii, center in enumerate(centers):
										is_arucotag_id_correct = True if(ids[ii] == 0) else False;
										is_tag_within_box      = True if( ((center[0] >= x1) and (center[0] <= x2)) and ((center[1] >= y1) and (center[1] <= y2)) ) else False;
										if(is_arucotag_id_correct and is_tag_within_box):
											is_detected_person_arucotagged = True
									if(is_detected_person_arucotagged):
										confidence         = np.ceil((box.cpu().conf[0]*100))/100   # box is on cuda before this, so cast it to .cpu(), this won't fail even when running on cpu
										current_detection  = np.array([[x1, y1, x2, y2, confidence]])
										yolo_detections    = np.vstack((yolo_detections, current_detection))
										current_loc        = [(x1 + x2) // 2,y2] # this can be replaced with average of feet positions
										current_cord       = cv2.perspectiveTransform(np.array([[current_loc]], dtype=np.float32), self.homography_matrix)
										current_cordx      = current_cord[0][0][0]
										current_cordy      = current_cord[0][0][1]
										timestamp          = time.time()
										timestamp_ordered  = datetime.datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")
										w, h               = x2 - x1, y2 - y1
										cvzone.cornerRect(cv_img, (x1, y1, w, h),l=9,rt=2,colorR=(0,255,0))
										self.loc_xy_file.write(timestamp_ordered+","+str(current_cordx)+","+str(current_cordy)+"\n");
										self.label_cordx.setText(str(np.round(1000*current_cordx)/1000)) # to fix the number of decimal points, otherwise the display jitters
										self.label_cordy.setText(str(np.round(1000*current_cordy)/1000)) # to fix the number of decimal points, otherwise the display jitters
									else:
										w, h = x2 - x1, y2 - y1
										cvzone.cornerRect(cv_img, (x1, y1, w, h),l=9,rt=2,colorR=(0,0,255))
								else:
									continue # skip this box

							## disabling SORT functionality for now, we'll use aruco tags to identify the person
							#result_tracker = self.tracker.update(yolo_detections)
							#for res in result_tracker:
							#	x1, y1, x2, y2, id = res	
							#	x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
							#	w, h = x2 - x1, y2 - y1
							#	cvzone.putTextRect(cv_img, f'ID: {id}', (x1, y1), scale=1,thickness=1,colorR=(0,0,0))
							#	cvzone.cornerRect(cv_img, (x1, y1, w, h),l=9,rt=1,colorR=(0,0,0))
							#	# feet position markers can come here if we use pose estimation eventually
							
						self.change_pixmap_signal.emit(cv_img)
						#time.sleep(0.1)
				# shut down capture system
				self.cap_obj.release()

			def stop(self):
				"""Sets run flag to False and waits for thread to finish"""
				self._run_flag = False
				self.loc_xy_file.close();
				self.wait()

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
				self.frame = ClickableFrameQLabel(self.centralwidget) # the picture frame is created separately 
				self.frame.setGeometry(QRect(10, 10, 1280, 720))

				# to avoid premature exit warnings
				self.cv2thread = VideoThread(cap, self.label_cordx, self.label_cordy) 
				self.cv2thread.change_pixmap_signal.connect(self.update_image)
				self.cv2thread.start();

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
		cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # this should match what's inside calibration_gui.py
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(lines[1]))
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(lines[2]))
		actual_cam_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		cam_ratio  = float(min((1280,720)))/min(actual_cam_res) # 1280x720 is the interface size
		
		if((actual_cam_res[0] != int(lines[1])) and ((actual_cam_res[1] != int(lines[2])))):
			print("The cam could not obtain the desired resolution for some reason, debug this before continuing")
		else:
			# if everything's OK, start GUI
			app = QtWidgets.QApplication(sys.argv)
			w = MainWindow(cap=cap)
			w.show()
			sys.exit(app.exec_())
	else:
		print("exiting")


