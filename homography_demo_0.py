# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import time
from typing import List
import glob
from math import sqrt

# from RealsenseManager import *
# from ConsolePrinter import Printer
#
#
src_points = np.array([
	[0., 0.],
	[10., 0.],
	[9., 6.],
	[3., 6.],
], dtype=np.float32)

dst_points = np.array([
	[0., 0.],
	[10., 0.],
	[10., 8.],
	[0., 8.],
], dtype=np.float32)


# PerspectiveMatrix = cv2.getPerspectiveTransform(src_points, dst_points)
#
pts = np.array([[5.5, 4.5, 1], [9., 6., 1]], dtype=np.float32)
#
# dst = cv2.perspectiveTransform(pts, PerspectiveMatrix)
#


# src_points = np.array([[165., 270.], [835., 270.], [360., 125.], [615., 125.]], dtype="float32")
# dst_points = np.array([[165., 270.], [835., 270.], [165., 30.], [835., 30.]], dtype="float32")

M = cv2.getPerspectiveTransform(src_points, dst_points)

print(M)

# dst = cv2.perspectiveTransform(pts, M)
dst = np.matmul(pts, M)

print('dst=', dst)
