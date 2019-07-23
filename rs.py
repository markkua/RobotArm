# -*- encoding: utf-8 -*-

import pyrealsense2 as rs
import cv2


def test_camera():
	color_map = rs.colorizer()
	
	pipe = rs.pipeline()
	
	pipe.start()
	print("pipe started")
	
	printer = rs.rates_printer()
	
	data = pipe.wait_for_frames().apply_filter(printer).apply_filter(color_map)
	
	cv2.imshow("Camera", data)
	

if __name__ == '__main__':
	test_camera()