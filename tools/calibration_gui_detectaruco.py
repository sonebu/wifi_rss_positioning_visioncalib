import cv2 
import numpy as np

# to label the markers on the video frames
colors = [(255, 0, 0),  # Blue topLeft
          (0, 255, 0),  # Green topRight
          (0, 0, 255),  # Red bottomRight
          (255, 255, 0)] # Cyan bottomLeft

## left element of tuple is the marker ID, right element is the corresponding marker center cord in 2D
## made a somewhat random coice of grid coordinates but it's actually doing a polar sweep (kind of)
aruco_markerID_centercords = [ (0,  [0 , 0]),
						 	   (1,  [1 , 0]),
						 	   (2,  [1 , 1]),
						 	   (3,  [0 , 1]),
						 	   (4,  [2 , 0]),
						 	   (5,  [2 , 2]),
						 	   (6,  [0 , 2]),
						 	   (7,  [3 , 3]),
						 	   (8,  [4 , 0]),
						 	   (9,  [4 , 4]),
						 	   (10, [0 , 4]),
						 	   (11, [7 , 0]),
						 	   (12, [7 , 7]),
						 	   (13, [0 , 7]),
						 	   (14, [10, 0]),
						 	   (15, [10, 10]),
						 	   (16, [0 , 10]) ] 

##  The mapping looks like this (corners are aruco tags, numbers show marker IDs):
## 
##     0---1---4-------8-----------11----------14---
##     |   |   |   |   |   |   |   |   |   |   |   |
##     3---2----------------------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     6-------5------------------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     ------------7--------------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     10--------------9----------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     ---------------------------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     ---------------------------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     13--------------------------12---------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     ---------------------------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     ---------------------------------------------
##     |   |   |   |   |   |   |   |   |   |   |   |
##     16--------------------------------------15---
##     |   |   |   |   |   |   |   |   |   |   |   |
##     ---------------------------------------------

arucoDict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
arucoParams   = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

def aruco_detect_draw(frame, verbose=False, draw=True):
	aruco_out = arucoDetector.detectMarkers(frame)

	(corners_raw, ids_raw, rejected) = aruco_out
	centers = []
	corners = []
	ids     = []
	# verify *at least* one ArUco marker was detected
	if len(corners_raw) > 0:
		# flatten the ArUco IDs list
		ids_raw = ids_raw.flatten()

		# filter out irrelevant IDs (0-16 is for calibration, 17 is for person tagging)
		for ii, _ in enumerate(corners_raw):
			if(ids_raw[ii] <= 17):
				ids.append(ids_raw[ii])
				corners.append(corners_raw[ii])

		ids = np.array(ids)
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			if(draw):
				frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
			(topLeft, topRight, bottomRight, bottomLeft) = markerCorner.reshape((4, 2))

			# convert each of the (x, y)-coordinate pairs to integers
			topRight    = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft  = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft     = (int(topLeft[0]), int(topLeft[1]))

			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
			cX = int((topRight[0] + topLeft[0] + bottomRight[0] + bottomLeft[0]) / 4.0)
			cY = int((topRight[1] + topLeft[1] + bottomRight[1] + bottomLeft[1]) / 4.0)
			centers.append((cX, cY));
			
			### Commented to use the drawDetectedMarkers() function
			#frame = cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
			# draw the bounding box of the ArUCo detection
			#cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			#cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			#cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			#cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
			#cv2.circle(frame, topLeft, 5, colors[0], -1)  # Blue dot
			#cv2.circle(frame, topRight, 5, colors[1], -1)  # Green dot
			#cv2.circle(frame, bottomRight, 5, colors[2], -1)  # Red dot
			#cv2.circle(frame, bottomLeft, 5, colors[3], -1)  # Cyan dot
			#if(verbose):
			#	print("BLUE ", topLeft)
			#	print("GREEN ", topRight)
			#	print("RED ", bottomRight)
			#	print("CYAN ", bottomLeft)
			# draw the ArUco marker ID on the frame
			#cv2.putText(frame, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			#if(verbose):
			#	print(markerID, markerCorner)
	else:
		pass # this is not really an error
		#print("[ERROR]: No aruco tags detected")

	return centers, corners, ids, frame
