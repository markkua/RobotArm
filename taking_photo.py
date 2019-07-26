# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import time
import os
from typing import List
import pyrealsense2 as rs


import pyrealsense2 as rs
import numpy as np
import cv2


class DatasetCapture:
	def __init__(self):
		# Configure depth and color streams
		self.pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
		
		# Start streaming
		self.pipeline.start(config)
	
		# Configure depth and color streams
		self.pipeline = rs.pipeline()
		config = rs.config()
		config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
		config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

		# Start streaming
		self.pipeline.start(config)

	def start_photograph(self):
		try:
			while True:
		
				# Wait for a coherent pair of frames: depth and color
				frames = self.pipeline.wait_for_frames()
				depth_frame = frames.get_depth_frame()
				color_frame = frames.get_color_frame()
				if not depth_frame or not color_frame:
					continue
		
				# Convert images to numpy arrays
				depth_image = np.asanyarray(depth_frame.get_data())
				color_image = np.asanyarray(color_frame.get_data())
		
				# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
				depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
		
				# Stack both images horizontally
				# images = np.hstack((color_image, depth_colormap))
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
					path = "dataset/"
					filename = path + int(time.time()).__str__() + '.jpg'
					cv2.imwrite(filename, images)
					print('\033[32m image saved at %s！\033[0m' % filename)
		finally:
		
			# Stop streaming
			self.pipeline.stop()
			cv2.destroyAllWindows()
		

if __name__ == '__main__':
	capture = DatasetCapture()
	capture.start_photograph()
