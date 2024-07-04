import cv2 

colors = [(255, 0, 0),  # Blue
          (0, 255, 0),  # Green
          (0, 0, 255),  # Red
          (255, 255, 0)] # Cyan

arucoDict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
arucoParams   = cv2.aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

def aruco_detect_draw(frame, verbose=False):
	aruco_out = arucoDetector.detectMarkers(frame)

	(corners, ids, rejected) = aruco_out

	# verify *at least* one ArUco marker was detected
	if len(corners) > 0:
		# flatten the ArUco IDs list
		ids = ids.flatten()

		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned
			# in top-left, top-right, bottom-right, and bottom-left
			# order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# draw the bounding box of the ArUCo detection
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

			cv2.circle(frame, topLeft, 5, colors[0], -1)  # Blue dot
			cv2.circle(frame, topRight, 5, colors[1], -1)  # Green dot
			cv2.circle(frame, bottomRight, 5, colors[2], -1)  # Red dot
			cv2.circle(frame, bottomLeft, 5, colors[3], -1)  # Cyan dot
			if(verbose):
				print("BLUE ", topLeft)
				print("GREEN ", topRight)
				print("RED ", bottomRight)
				print("CYAN ", bottomLeft)
			
			# compute and draw the center (x, y)-coordinates of the
			# ArUco marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)

			# draw the ArUco marker ID on the frame
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			if(verbose):
				print(markerID,markerCorner)
	else:
		print("[ERROR]: No aruco tags detected")

	return aruco_out, frame
