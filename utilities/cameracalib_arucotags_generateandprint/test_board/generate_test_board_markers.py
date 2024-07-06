#!/usr/bin/env python

import cv2, argparse
import numpy as np

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36h12)
for i in range(17):
	tag = np.zeros((300, 300, 1), dtype="uint8")
	cv2.aruco.generateImageMarker(arucoDict, i, 300, tag, 1)
	cv2.imwrite("out"+str(i)+".png", tag)
