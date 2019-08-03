# -*- encoding: utf-8 -*-


import cv2
import numpy as np
from math import pi, exp
import time


class ImgProcessor:
	
	@staticmethod
	def trans_2_binary(image, threshold):
		t, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
		return binary
	
	class HSV_Color:
		red = []
	
	@classmethod
	def _pick_color(cls, image, threshold, miu: np.array, sig: np.array):
		
		hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		
		hsv_image = np.asarray(hsv_image)

		mask = np.zeros(hsv_image.shape, dtype=np.uint8)

		signeg = np.linalg.inv(sig)
		c = 1 / (2 * pi)**1.5 / np.linalg.det(sig)**0.5
		
		# 置信区间
		large_miu = np.asarray([miu[0]*180, miu[1]*255, miu[2]*255])
		# large_sigma = np.asarray([sig[0, 0] * 180, sig[1, 1] * 255, sig[2, 2] * 255]) * 3
		
		# 判断是否在 置信区间内
		def if_in_confidence(x):
			width = 10
			if not large_miu[0] - width < x[0] < large_miu[0] + width:
				return False
			return True

		h, w, chanel = hsv_image.shape
		for i in range(h):
			for j in range(w):
				xi = np.asarray(hsv_image[i][j])
				
				# 粗筛选
				if not if_in_confidence(xi):
					mask[i][j] = [0, 0, 0]
					# print("skip")
					continue
				
				xi = np.asarray([xi[0] / 180, xi[1] / 255, xi[2] / 255])
				temp1 = np.transpose(xi - miu)
				temp2 = np.matmul(temp1, signeg)
				temp3 = np.matmul(temp2, (xi - miu))
				upper = temp3 / -2

				pixi = c * exp(upper)
				# print("i=%d, j=%d" % (i, j))
				# print('xi=', xi)
				# print(temp1)
				# print(temp2)
				# print(temp3)
				# print('upper=', upper)
				# print('c=', c)
				# print('pixi=', pixi)
				if pixi > threshold:
					# mask[i][j] = [255, 255, 255]
					mask[i][j] = [255, 255, 255]
				else:
					mask[i][j] = [0, 0, 0]
		return mask
	
	@classmethod
	def pick_red(cls, image, threshold):
		image = resize(image, 0.1)
		
		miu = np.asarray([0.957300173750032, 0.702467503221102, 0.519036449382812])
		sig = np.asarray([[0.021501230242404, -8.839793231638757e-04, 0.007253323195827],
		                    [-8.839793231638757e-04, 0.003226414505887, -0.001972126596248],
		                    [0.007253323195827, -0.001972126596248, 0.006512052763508]])
		return cls._pick_color(image, threshold, miu, sig)
	
	# @classmethod
	# def pick_green(cls, image, threshold):
	# 	image = cls.resize(image, 0.1)
	#
	# 	miu = np.asarray( [0.230604265949190, 0.528880344417954, 0.477433844165268])
	# 	sig = np.asarray([[8.142333919138685e-05,-3.528365300776935e-04,2.419821012445137e-05],
	# 	                  [-3.528365300776935e-04,0.004714512541749,-0.001336533557575],
	# 	                  [2.419821012445137e-05,-0.001336533557575,0.003038826607760]])
	#
	# 	result = cls._pick_color(image, threshold, miu, sig)
	# 	return result
	#


def resize(image, percentage):
	w, h, c = image.shape
	return cv2.resize(image, (int(h*percentage), int(w*percentage)))


def get_mass_center_float(gray_img):
	result = cv2.moments(gray_img, 1)
	if not 0 == result['m00']:
		x = round(result['m10'] / result['m00'])
		y = round(result['m01'] / result['m00'])
	else:
		x, y = 0, 0
	return np.asarray([x, y])


def test_hsv2rgb():
	hsv = np.asarray([60, 180, 180])
	image = np.zeros([100, 100, 3], np.uint8)
	for i in range(100):
		for j in range(100):
			image[i][j] = hsv
	rgb_image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
	cv2.imshow('rgb', rgb_image)
	cv2.waitKey(0)


def test_rgb2hsv():
	rgb = np.asarray([0, 255, 0])
	image = np.zeros([100, 100, 3], np.uint8)
	for i in range(100):
		for j in range(100):
			image[i][j] = rgb
	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	print(hsv_image[0, 0])
	

if __name__ == '__main__':
	image = cv2.imread("old_py/ControlPoint/green/001.png")
	start_time = time.time()
	# result = ImgProcessor.pick_red(image, 0.8)
	result = ImgProcessor.pick_green(image, 0.8)
	print("time till pick color=", time.time()-start_time)
	
	canny = cv2.Canny(result, 100, 150)
	canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
	
	result = np.hstack((canny, ImgProcessor.resize(image, 0.1)))
	
	print("time till canny=", time.time() - start_time)
	
	cv2.imshow("0.8", result)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	# test_hsv2rgb()
	# test_rgb2hsv()


