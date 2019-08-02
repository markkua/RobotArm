# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import time
import os
from typing import List
import pyrealsense2 as rs

from RealsenseManager import *


class DatasetCapture:
	def __init__(self):
		self.realsense = RealsenseManager()

	def start_photograph(self, path: str):
		count = 1
		try:
			while True:
		
				# Wait for a coherent pair of frames: depth and color
				frames = self.realsense.get_aligned_frames()
				color_frame = frames.get_color_frame()
				if not color_frame:
					continue
		
				# Convert images to numpy arrays
				color_image = np.asanyarray(color_frame.get_data())

				# Stack both images horizontally
				images = color_image
		
				# Show images
				cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
				cv2.imshow('RealSense', images)
				key = cv2.waitKey(1)
				# Press esc or 'q' to close the image window
				if key & 0xFF == ord('q') or key == 27:
					cv2.destroyAllWindows()
					break
				elif key == ord('s') or key == ord(' '):
					filename = path + "%03d" % count + '.png'
					count += 1
					cv2.imwrite(filename, images)
					print('\033[32m image saved at %s\033[0m' % filename)
		finally:
			cv2.destroyAllWindows()
		

if __name__ == '__main__':
	capture = DatasetCapture()
	path = "./ControlPoint/green/"
	print("working dir: ", path)
	capture.start_photograph(path=path)
	# capture.start_photograph(path="data/")
