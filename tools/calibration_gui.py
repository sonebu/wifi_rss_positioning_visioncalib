################################
### Calibration PyQt5 GUI
###
### Valid states: enter_exp, enter_devstring, select_mode, frame_selection, waiting4validframe, waiting4calib, uncalibrated_pt

### The layout is designed in Qt5 Designer 5.15, the class definition is imported from there 
from calibration_gui_qt5designer import Ui_MainWindow

### Functional part of the Calibration GUI is in this file (actions, state machine etc.)
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRect, QPoint
import sys, os, cv2, time
import numpy as np
from calibration_gui_detectaruco import aruco_detect_draw, aruco_markerID_centercords

calibpts   = []
calibcords = []
testpt     = None
testcord   = None
calibmode  = None # will hold "ArUco" or "Manual"
state      = None # will be initialized inside MainWindow init
hmg_mtx    = None # homography matrix
hmg_exists = False # flag denoting whether a computed homography matrix exists at any given moment

### The interface has a fixed size 1280x720 video window
### The desired cam resolution (if it matches the real output the camera can provide) will be resized to that without aspect ratio distortion
### You need to check the available camera formats, choose a backend and a fourcc format string 
desired_cam_res     = (1920,1080)
desired_cam_fmt     = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

# adapted from: https://gist.github.com/docPhil99/ca4da12c9d6f29b9cea137b617c7b8b1
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, cap_obj):
        super().__init__()
        self._run_flag = True
        self.cap_obj = cap_obj;
        self.cv_img_memory    = None;

    def run(self):
        global calibpts, calibcords, hmg_exists, cam_ratio
        while self._run_flag:
            ret, cv_img = self.cap_obj.read()
            if ret:
                if(state == "waiting4validframe"):
                    cv_img_tmp = self.cv_img_memory.copy()
                    if(calibmode == "ArUco"):
                        (_, _, _, _) = aruco_detect_draw(cv_img_tmp, verbose=False, draw=True)
                        self.change_pixmap_signal.emit(cv_img_tmp)
                    else:
                        self.change_pixmap_signal.emit(self.cv_img_memory)
                    time.sleep(0.1)
                elif((state == "waiting4calib") or (state == "uncalibrated_pt")):
                    cv_img_tmp = self.cv_img_memory.copy()
                    if(calibmode == "Manual"):
                        if(len(calibpts) > 0):
                            for i, pt in enumerate(calibpts):
                                cv_img_tmp = cv2.circle(cv_img_tmp, (int(pt.x()/cam_ratio), int(pt.y()/cam_ratio)), 10, 
                                                                (0,255,0) if calibcords[i] is not None else (0,0,255), thickness=2, lineType=8, shift=0)
                                cv_img_tmp = cv2.putText(cv_img_tmp, str(calibcords[i]) if calibcords[i] is not None else "enter calib coords", 
                                                                 (int(pt.x()/cam_ratio)-15, int(pt.y()/cam_ratio)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0) if calibcords[i] is not None else (0,0,255), 1, cv2.LINE_AA)
                            if(testpt is not None):
                                cv_img_tmp = cv2.circle(cv_img_tmp, (int(testpt[0]/cam_ratio), int(testpt[1]/cam_ratio)), 10, (255,0,0), thickness=2, lineType=8, shift=0)
                                cv_img_tmp = cv2.putText(cv_img_tmp, "("+str(testcord[0])+","+str(testcord[1])+")", (int(testpt[0]/cam_ratio)-15, int(testpt[1]/cam_ratio)-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1, cv2.LINE_AA)

                            self.change_pixmap_signal.emit(cv_img_tmp)
                        else:
                            self.change_pixmap_signal.emit(self.cv_img_memory)
                        time.sleep(0.1)
                    elif(calibmode == "ArUco"):
                        (aruco_centers, aruco_corners, aruco_ids, cv_img_tmp) = aruco_detect_draw(cv_img_tmp, verbose=False, draw=True)
                        calibpts = aruco_centers
                        aruco_marker_center_cords = []
                        for idx in aruco_ids:
                            aruco_marker_center_cords.append(aruco_markerID_centercords[idx][1]) # 0 is ID, 1 is x,y coordinate
                        calibcords = aruco_marker_center_cords
                        if(testpt is not None):
                                cv_img_tmp = cv2.circle(cv_img_tmp, (int(testpt[0]), int(testpt[1])), 10, (0,0,255), thickness=2, lineType=8, shift=0)
                                cv_img_tmp = cv2.putText(cv_img_tmp, "("+str(testcord[0])+","+str(testcord[1])+")", (int(testpt[0])-15, int(testpt[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                        self.change_pixmap_signal.emit(cv_img_tmp)
                        time.sleep(0.1)
                    else:
                        self.change_pixmap_signal.emit(self.cv_img_memory)
                else:
                    self.cv_img_memory = cv_img.copy();
                    if(calibmode == "ArUco"):
                        # ArUco detection and marking on frame (feed marked image if aruco is selected)
                        (_, _, _, cv_img) = aruco_detect_draw(cv_img, verbose=False, draw=True);
                        self.change_pixmap_signal.emit(cv_img)
                    else:
                        self.change_pixmap_signal.emit(self.cv_img_memory) # feed unmarked image if manual mode is selected or no mode is selected yet

            # time.sleep(0.1) # enable this if there's too much flicker on the GUI

        # shut down capture system
        self.cap_obj.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class ClickableFrameQLabel(QLabel):
    clickpos = pyqtSignal(QPoint)
    def mousePressEvent(self, event):
        self.clickpos.emit(event.pos())
        QLabel.mousePressEvent(self, event)

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)

        # Get the layout up
        self.setupUi(self)
        self.frame = ClickableFrameQLabel(self.centralwidget) # the picture frame is created separately 
        self.frame.setGeometry(QRect(35, 120, 1280, 720))
        self.cv2thread = None       # to avoid premature exit warnings
        self.homography_mode = None # no default mode, user has to choose something

        # Make the connections
        self.pushButton.clicked.connect(self.qpb_expname_clicked)
        self.pushButton_2.clicked.connect(self.qpb_devstring_clicked)
        self.pushButton_3.clicked.connect(self.qpb_calibsubmit_clicked)
        self.pushButton_4.clicked.connect(self.qpb_shootframe_clicked)
        self.pushButton_5.clicked.connect(self.qpb_cancelframe_clicked)
        self.pushButton_6.clicked.connect(self.qpb_validframe_clicked)
        self.pushButton_7.clicked.connect(self.qpb_gethomography_clicked)
        self.pushButton_8.clicked.connect(self.qpb_testptsubmit_clicked)
        self.pushButton_9.clicked.connect(self.qpb_mode_clicked)
        self.pushButton_10.clicked.connect(self.qpb_calibdelete_clicked)

        # Set up initial state (starting from experiment entry)
        global state
        state = "enter_exp"
        self.lineEdit_2.setEnabled(False);
        self.lineEdit_3.setEnabled(False);
        self.lineEdit_4.setEnabled(False);
        self.listWidget.item(0).setFlags(self.listWidget.item(0).flags() & ~Qt.ItemIsSelectable); # | Qt.ItemIsSelectable to turn it back on
        self.listWidget.item(1).setFlags(self.listWidget.item(1).flags() & ~Qt.ItemIsSelectable); # | Qt.ItemIsSelectable to turn it back on
        self.listWidget.setEnabled(False);
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        self.pushButton_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.pushButton_6.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        self.pushButton_9.setEnabled(False)
        self.pushButton_10.setEnabled(False)

        # some buffers
        self.cv_img_buffer = None
        self.validated_exp_path = None

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

    ##################################################################
    # callbacks
    def qpb_expname_clicked(self):
        submitted_exp_name = self.lineEdit.text()
        # check if the submitted experiment foldername submitted contains only alphanumeric chars and _ 
        for c in submitted_exp_name:
            if c.isalnum() or c in ['_']:
                continue
            else:
                self.label.setText("<font color='red'>Only alnum chars and _</font>")
                return # dummy return to get out of the function

        # check if the submitted experiment foldername exists already
        if(os.path.isdir("../experiments/" + submitted_exp_name)):
            self.label.setText("<font color='red'>Exp exists, cant overwrite</font>") 
            return # dummy return to get out of the function
        else:
            self.validated_exp_path = "../experiments/" + submitted_exp_name
            os.mkdir(self.validated_exp_path)
            # continue, this part is done
            self.label.setText("<font color='green'>Created exp with submitted name</font>")
            self.lineEdit.setEnabled(False);
            self.lineEdit_2.setEnabled(True);
            self.lineEdit.setStyleSheet("QLineEdit { background-color: gray; font-weight: bold}")
            global state
            state = "enter_devstring"
            self.pushButton.setEnabled(False)
            self.pushButton_2.setEnabled(True)

    def qpb_devstring_clicked(self):
        submitted_devstring = self.lineEdit_2.text()
        # check if the submitted devstring submitted contains only alphanumeric chars, _ and / 
        for c in submitted_devstring:
            if c.isalnum() or c in ['_'] or c in ['/']:
                continue
            else:
                self.label.setText("<font color='red'>Invalid devstring</font>")
                return # dummy return to get out of the function

        # check if the submitted devstring can be opened as a camera
        cap = cv2.VideoCapture(submitted_devstring)
        if not cap.isOpened():
            self.label_2.setText("<font color='red'>Cam cannot be opened</font>")
            return # dummy return to get out of the function
        else:
            self.label_2.setText("<font color='green'>Cam opened, streaming</font>")
            self.lineEdit_2.setEnabled(False);
            self.lineEdit_2.setStyleSheet("QLineEdit { background-color: gray; font-weight: bold}")
            cap.set(cv2.CAP_PROP_FOURCC, desired_cam_fmt)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_cam_res[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_cam_res[1])
            actual_cam_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            global cam_ratio
            cam_ratio  = float(min((1280,720)))/min(actual_cam_res) # 1280x720 is the interface size
            with open(self.validated_exp_path + "/devstring.txt","w") as f:
                f.write(self.lineEdit_2.text()+"\n")
                f.write(str(actual_cam_res[0])+"\n")
                f.write(str(actual_cam_res[1]))
            global state
            state = "select_mode"
            self.cv2thread = VideoThread(cap)
            self.cv2thread.change_pixmap_signal.connect(self.update_image)
            self.cv2thread.start();
            self.listWidget.item(0).setFlags(self.listWidget.item(0).flags() | Qt.ItemIsSelectable);
            self.listWidget.item(1).setFlags(self.listWidget.item(1).flags() | Qt.ItemIsSelectable);
            self.listWidget.setEnabled(True);
            self.pushButton_2.setEnabled(False)
            self.pushButton_9.setEnabled(True)

    def qpb_mode_clicked(self):
        global state, calibmode
        if(state == "select_mode"):
            if(len(self.listWidget.selectedItems()) > 0):
                self.homography_mode = self.listWidget.selectedItems()[0].text();
                self.label_7.setText("Mode: "+self.homography_mode)
                self.listWidget.setStyleSheet("QListWidget { background-color: gray; font-weight: bold}");
                state = "frame_selection"
                self.frame.clickpos.connect(self.ql_frame_clicked)
                self.listWidget.setEnabled(False)
                self.pushButton_9.setEnabled(False)
                self.pushButton_4.setEnabled(True)
                calibmode = self.homography_mode
            else:
                pass # the user didnt choose anything yet, dont continue
        else:
            pass # do nothing if the state is not right

    def qpb_shootframe_clicked(self):
        global state
        if(state == "frame_selection"):
            state = "waiting4validframe"
            self.pushButton_4.setEnabled(False)
            self.pushButton_5.setEnabled(True)
            self.pushButton_6.setEnabled(True)

    def qpb_cancelframe_clicked(self):
        global state
        if(state == "waiting4validframe"):
            state = "frame_selection"
            self.pushButton_4.setEnabled(True)
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setEnabled(False)

    def qpb_validframe_clicked(self):
        global state, calibmode, calibpts, hmg_exists
        if(state == "waiting4validframe"):
            state = "waiting4calib"
            time.sleep(1.0) # wait for an update on the calibpts array (this is a horrible way of doing this though, we should reconsider when we have the time)
            self.pushButton_4.setEnabled(False)
            self.pushButton_5.setEnabled(False)
            self.pushButton_6.setStyleSheet('QPushButton {background-color: green;}')
            self.pushButton_6.setEnabled(False)
            cv2.imwrite(self.validated_exp_path + "/reference_image.png", self.cv_img_buffer)
            if(calibmode == "ArUco"):
                # enable homography calculation and testing (if possible)
                if(len(calibpts) >= 4):
                    self.pushButton_7.setEnabled(True) # enable homography calc
                    if(hmg_exists):
                        self.pushButton_8.setEnabled(True) # enable testing with current homography
                        self.lineEdit_4.setEnabled(True)  # same as above
                else:
                    self.pushButton_7.setEnabled(False) # disable homography calc, not enough pts
                    self.pushButton_8.setEnabled(False) # disable testing with current homography
                    self.lineEdit_4.setEnabled(False)   # same as above

    @pyqtSlot(QPoint)
    def ql_frame_clicked(self, pointpos):
        global calibpts, calibcords, state
        if(calibmode == "Manual"):
            if(state == "waiting4calib"):
                calibpts.append(pointpos); # add clicked location to list of calibration points
                calibcords.append(None);   # new point, uncalibrated
                state = "uncalibrated_pt"
                self.lineEdit_3.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                self.pushButton_10.setEnabled(True)
        else:
            if(state == "waiting4calib"):
                pass 
                ### Dropped functionality:
                # looks at aruco tag locations and chooses closest aruco tag based on clicked location
                # the chosen aruco tag becomes a new calibpt-calibcord duo, uncalibrated
                # the calibration mechanism follows the same path as the manual from hereon
                # the aruco tag points are thus shown in red during rendering, stays that way when rendering stops
                # when you click them they request calib coords (no extra drawing, just puttext), 
                # when you enter calib cord in turns green with relevant coord (colors handled in cv2 loop)
            else:
                pass # do nothing if the state is not right

    def qpb_calibsubmit_clicked(self):
        global calibcords, state, hmg_exists, testpt, testcord
        if((state == "waiting4calib") or (state == "uncalibrated_pt")):
            if(len(calibpts)>0):
                submitted_tuple_aslist = self.lineEdit_3.text().replace("(","").replace(")","").split(",")
                input_health = True
                if(len(submitted_tuple_aslist) != 2):
                    input_health = False
                if((not (submitted_tuple_aslist[0].isdigit())) or (not (submitted_tuple_aslist[1].isdigit()))):
                    input_health = False
                if(input_health):
                    calibcords[-1] = (int(submitted_tuple_aslist[0]),int(submitted_tuple_aslist[1]))
                    state = "waiting4calib"
                    if(len(calibpts) >= 4):
                        self.pushButton_7.setEnabled(True) # enable homography calc
                        if(hmg_exists):
                            self.pushButton_8.setEnabled(True) # enable testing with current homography
                            self.lineEdit_4.setEnabled(True)   # same as above
                    else:
                        self.pushButton_7.setEnabled(False) # disable homography calc, not enough pts
                        self.pushButton_8.setEnabled(False) # disable testing with current homography
                        self.lineEdit_4.setEnabled(False)   # same as above
                        testpt = None
                        testcord = None

    def qpb_calibdelete_clicked(self):
        global calibpts, calibcords, state, hmg_exists, testpt, testcord
        if((state == "waiting4calib") or (state == "uncalibrated_pt")):
            if(len(calibpts)>0):
                calibpts.pop() # delete last element
                calibcords.pop() # delete last element
            # after the pop, update state machine
            if(len(calibpts)==0):
                state = "waiting4calib"
            elif(len(calibpts)>0):
                if(calibcords[-1] is not None):
                    state = "waiting4calib"
                else:
                    state = "uncalibrated_pt"
                if(len(calibpts) >= 4):
                    self.pushButton_7.setEnabled(True) # enable homography calc
                    if(hmg_exists):
                        self.pushButton_8.setEnabled(True) # enable testing with current homography
                        self.lineEdit_4.setEnabled(True)  # same as above
                else:
                    self.pushButton_7.setEnabled(False) # disable homography calc, not enough pts
                    self.pushButton_8.setEnabled(False) # disable testing with current homography
                    self.lineEdit_4.setEnabled(False)   # same as above
                    testpt = None
                    testcord = None

    def qpb_gethomography_clicked(self):
        global calibpts, calibcords, calibmode, hmg_mtx, hmg_exists
        # assume that the other buttons track the state of this guy
        pts_arr = np.zeros((len(calibpts),2))
        for i, qpt in enumerate(calibpts):
            pts_arr[i,0] = qpt.x() if calibmode == "Manual" else qpt[0]
            pts_arr[i,1] = qpt.y() if calibmode == "Manual" else qpt[1]
        crd_arr = np.zeros((len(calibpts),2))
        for i, qcr in enumerate(calibcords):
            crd_arr[i,0] = qcr[0]
            crd_arr[i,1] = qcr[1]
        hmg_mtx, _ = cv2.findHomography(pts_arr, crd_arr)
        cv2.imwrite(self.validated_exp_path + "/reference_image_withHomographyAnchors.png", self.cv_img_buffer)
        np.save(self.validated_exp_path + "/homography_matrix.npy", hmg_mtx)
        np.savetxt(self.validated_exp_path + "/homography_matrix.txt", hmg_mtx)
        print("")
        print("Homography Matrix:")
        print(hmg_mtx)
        print("")
        hmg_exists = True
        if(len(calibpts) >= 4):
            if(hmg_exists):
                self.pushButton_8.setEnabled(True) # enable testing with current homography
                self.lineEdit_4.setEnabled(True)  # same as above
        else:
            self.pushButton_8.setEnabled(False) # disable testing with current homography
            self.lineEdit_4.setEnabled(False)   # same as above

    def qpb_testptsubmit_clicked(self):
        global hmg_mtx, testpt, testcord
        # assume that the other buttons track the state of this guy
        submitted_tuple_aslist = self.lineEdit_4.text().replace("(","").replace(")","").split(",")
        input_health = True
        if(len(submitted_tuple_aslist) != 2):
            input_health = False
        if((not (submitted_tuple_aslist[0].isdigit())) or (not (submitted_tuple_aslist[1].isdigit()))):
            input_health = False
        if(input_health):
            testcord = np.array( [ [int(submitted_tuple_aslist[0]),int(submitted_tuple_aslist[1])] ] , dtype=np.float32) # extra dimensions seem necessary due to opencv's implementation
            hmg_mtx_inv = np.linalg.pinv(hmg_mtx)
            testpt   = cv2.perspectiveTransform(np.array([testcord]), hmg_mtx_inv)
            testpt   = testpt[0][0].astype(int)
            testcord = testcord[0].astype(int)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Q:
            self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
