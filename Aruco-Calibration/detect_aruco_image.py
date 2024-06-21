#!/usr/bin/env python

# import the necessary packages
import argparse
import imutils
import cv2
from cv2 import aruco
import sys
import numpy as np
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


cal3dPtLs = np.array([[27.25,5.25], [27.25,-5.25], [16.75,-5.25], [16.75, 5.25],
                       [5.25, 5.25], [5.25, -5.25], [-5.25, -5.25], [-5.25, 5.25],
                       [-18.25, 5.25], [-18.25, -5.25], [-28.75, -5.25], [-28.75, 5.25]], dtype=np.float32)


# load the input image from disk and resize it
print("[INFO] loading image...")
image = cv2.imread(args["image"])

cal2dPtLsid1 =  np.empty((0, 2), dtype=np.float32)
cal2dPtLsid3 =  np.empty((0, 2), dtype=np.float32)
cal2dPtLsid4 =  np.empty((0, 2), dtype=np.float32)
# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print("[INFO] ArUCo tag of '{}' is not supported".format(
		args["type"]))
	sys.exit(0)

# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()
detector = aruco.ArucoDetector(arucoDict, arucoParams)
(corners, ids, rejected) = detector.detectMarkers(image)
# verify *at least* one ArUco marker was detected
colors = [(255, 0, 0),  # Blue
          (0, 255, 0),  # Green
          (0, 0, 255),  # Red
          (255, 255, 0)] 
height, width, channels = image.shape

if len(corners) > 0:
	# flatten the ArUco IDs list
	ids = ids.flatten()

	# loop over the detected ArUCo corners
	for (markerCorner, markerID) in zip(corners, ids):
		# extract the marker corners (which are always returned in
		# top-left, top-right, bottom-right, and bottom-left order)
		
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		height, width, channels = image.shape
		
		# convert each of the (x, y)-coordinate pairs to integers
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))

		# draw the bounding box of the ArUCo detection
		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
		
		if markerID==1:
			cv2.circle(image, topLeft, 5, colors[0], -1)  # Blue dot
			print("BLUE ", topLeft)
			cal2dPtLsid1 = np.append(cal2dPtLsid1, [topLeft], axis=0)
			cv2.circle(image, topRight, 5, colors[1], -1)  # Green dot
			print("GREEN ", topRight)
			cal2dPtLsid1 = np.append(cal2dPtLsid1, [topRight], axis=0)
			cv2.circle(image, bottomRight, 5, colors[2], -1)  # Red dot
			print("RED ", bottomRight)
			cal2dPtLsid1 = np.append(cal2dPtLsid1, [bottomRight], axis=0)
			cv2.circle(image, bottomLeft, 5, colors[3], -1)  # Cyan dot
			print("CYAN ", bottomLeft)
			cal2dPtLsid1 = np.append(cal2dPtLsid1, [bottomLeft], axis=0)
		
		elif markerID==3:
			cv2.circle(image, topLeft, 5, colors[0], -1)  # Blue dot
			print("BLUE ", topLeft)
			cal2dPtLsid3 = np.append(cal2dPtLsid3, [topLeft], axis=0)
			cv2.circle(image, topRight, 5, colors[1], -1)  # Green dot
			print("GREEN ", topRight)
			cal2dPtLsid3 = np.append(cal2dPtLsid3, [topRight], axis=0)
			cv2.circle(image, bottomRight, 5, colors[2], -1)  # Red dot
			print("RED ", bottomRight)
			cal2dPtLsid3 = np.append(cal2dPtLsid3, [bottomRight], axis=0)
			cv2.circle(image, bottomLeft, 5, colors[3], -1)  # Cyan dot
			print("CYAN ", bottomLeft)
			cal2dPtLsid3 = np.append(cal2dPtLsid3, [bottomLeft], axis=0)
		elif markerID==4:
			cv2.circle(image, topLeft, 5, colors[0], -1)  # Blue dot
			print("BLUE ", topLeft)
			cal2dPtLsid4 = np.append(cal2dPtLsid4, [topLeft], axis=0)
			cv2.circle(image, topRight, 5, colors[1], -1)  # Green dot
			print("GREEN ", topRight)
			cal2dPtLsid4 = np.append(cal2dPtLsid4, [topRight], axis=0)
			cv2.circle(image, bottomRight, 5, colors[2], -1)  # Red dot
			print("RED ", bottomRight)
			cal2dPtLsid4 = np.append(cal2dPtLsid4, [bottomRight], axis=0)
			cv2.circle(image, bottomLeft, 5, colors[3], -1)  # Cyan dot
			print("CYAN ", bottomLeft)
			cal2dPtLsid4 = np.append(cal2dPtLsid4, [bottomLeft], axis=0)
		
		# compute and draw the center (x, y)-coordinates of the ArUco
		# marker
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

		# draw the ArUco marker ID on the image
		cv2.putText(image, str(markerID),
			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		print("[INFO] ArUco marker ID: {}".format(markerID))

		# show the output image
	height, width, channels = image.shape
	
	cv2.imshow("Image", image)
	cv2.waitKey(0)

cal2dPtLs= combined_array = np.vstack((cal2dPtLsid3, cal2dPtLsid1, cal2dPtLsid4))
homography_matrix, _ = cv2.findHomography(cal2dPtLs, cal3dPtLs)

cur_loc = [929, 663]
                
test_points = np.array([cur_loc], dtype=np.float32)

transformed_points = cv2.perspectiveTransform(np.array([test_points]), homography_matrix)

print(transformed_points)
cur_loc = [900, 536]
                
test_points = np.array([cur_loc], dtype=np.float32)

transformed_points = cv2.perspectiveTransform(np.array([test_points]), homography_matrix)
print(transformed_points)
cur_loc = [1034, 709]
                
test_points = np.array([cur_loc], dtype=np.float32)

transformed_points = cv2.perspectiveTransform(np.array([test_points]), homography_matrix)
print(transformed_points)
