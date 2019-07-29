# -*- encoding: utf-8 -*-


import cv2
import numpy as np


class ImgProcessor:
	
	@staticmethod
	def trans_2_binary(image, threshold):
		t, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
		return binary
	
	@staticmethod
	def pick_color(image, hue_center, hue_width):
		hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
		min_hue = hue_center - hue_width if hue_center - hue_width < 0 else 0
		max_hue = hue_center + hue_width if hue_center + hue_width > 255 else 255
		hsv = np.asarray([
			[min_hue, 0, 0],
			[max_hue, 255, 255]
		])
		
		# mask = cv2.inRange(hsv_image, lowerb=green_hsv[0], upperb=green_hsv[1])
		mask = cv2.inRange(hsv_image, lowerb=hsv[0], upperb=hsv[1])
		pick_image = cv2.bitwise_and(image, image, mask=mask)
		
		cv2.putText(image, "Image", (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255))
		cv2.putText(pick_image, "Hue ~ %d" % hue_center, (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255))
		
		# result = np.hstack((image, pick_image))
		
		return pick_image



