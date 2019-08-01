# -*- encoding: utf-8 -*-

from RealsenseManager import RealsenseManager
from ConsolePrinter import Printer
import ImgProcessor
import pyrealsense2 as rs

import cv2
from typing import List
import os
import sys
import numpy as np
from math import sin, cos, pi, sqrt, exp
import time


class Positioning:
	
	coor_filename = "data/control_points.txt"
	
	trans_para = np.zeros((6, 1), dtype=np.float)
	if_get_position = False  # 是否成功解算相机位置
	
	control_points = []  # [ [点名:str, [Robot坐标x2, y2, z2]] ]
	
	# 匹配到的控制点坐标 [ [点名:str, [Camera坐标x1, y1, z1], [Robot坐标x2, y2, z2]] ]
	_found_point_data = []
	
	def __init__(self, realsense: RealsenseManager):
		self.realsense = realsense
	
	def load_control_point_file(self, filename=None):
		if filename:
			fname = filename
		else:
			fname = self.coor_filename
		
		self.control_points.clear()
		
		file = open(fname)
		content_ls = file.readlines()
		for line in content_ls:
			if '' == line:
				continue
			line.replace(' ', '')
			pname, x, y, z = line.split(',')
			try:
				x = float(x)
				y = float(y)
				z = float(z)
				self.control_points.append([pname, np.asarray([x, y, z])])
			except ValueError as e:
				Printer.print("control point file error.", Printer.red)
				return False
		return True
		
	def get_camera_position(self, aligned_frames, threshold):
		# 检测目标点位置
		control_p_xy = self._match_control_points(aligned_frames, threshold)
		control_p_XYZ = []
		# TODO 转成相机空间坐标
		if not self._update_found_points(control_p_XYZ):
			return False
		if self._solve_7_paras():
			self.if_get_position = True
		else:
			self.if_get_position = False
		
	def coor_trans(self, xyz1: np.array):
		"""
		正向坐标变换
		:param xyz1: 相机坐标系的坐标xyz1
		:return: 返回转换后的np.array(xyz)， 若转换失败返回None
		"""
		if not self.if_get_position:
			Printer.print("No camera position, Cannot trans coor.", Printer.red)
			return None
		
		xyz1 = np.asarray(xyz1)
		if 3 != xyz1.size:
			Printer.print("coor trans error: xyz size wrong.", Printer.red)
			return None
		
		x1, y1, z1 = xyz1
		B = np.asarray([
			[1, 0, 0, 0, -z1, y1],
			[0, 1, 0, z1, 0, -x1],
			[0, 0, 1, -y1, x1, 0]
		])
		xyz2 = np.matmul(B, self.trans_para).reshape(3) + xyz1
		return xyz2
	 
	def _solve_7_paras(self) -> bool:
		n_pt = self._found_point_data.__len__()
		n_equ = n_pt * 3  # 方程数
		
		if n_equ < 6:  # 点数不够
			Printer.print("Control point not enough. Can't solve.", Printer.red)
			return False
		
		B = np.zeros((n_equ, 6), dtype=np.float)
		L = np.zeros((n_equ, 1), dtype=np.float)
		
		# 每个点对应三个方程
		for i in range(n_pt):
			p_name, xyz1, xyz2 = self._found_point_data[i]
			x1, y1, z1 = xyz1
			x2, y2, z2 = xyz2
			
			B[3 * i, :] = [1, 0, 0, 0, -z1, y1]
			B[3 * i + 1, :] = [0, 1, 0, z1, 0, -x1]
			B[3 * i + 2, :] = [0, 0, 1, -y1, x1, 0]
			
			L[3*i:3*i+3, 0] = [x2 - x1, y2 - y1, z2 - z1]
		
		# 解方程
		try:
			B = B * -1
			NBB = np.matmul(B.transpose(), B)
			NBB_inv = np.linalg.inv(NBB)
			BTPL = np.matmul(B.transpose(), L)
			result = np.matmul(NBB_inv, BTPL)
			self.trans_para[:, 0] = result[:, 0]
			
			V = np.matmul(B, result) - L
			r = n_equ - 6
			VTV = np.matmul(V.transpose(), V)
			sigma = sqrt(VTV[0, 0] / r)
			print('sigma=', sigma)
			if sigma > 0.1:
				Printer.print("sigma > 0.1", Printer.yellow)
				return False
			else:
				return True
		except Exception as e:
			Printer.print(e.__str__(), Printer.red)
			return False

	def _match_control_points(self, frames: rs.frame, threshold: float):
		"""
		匹配目标点
		:param frames:
		:return:  [ [点名:str, [图像坐标x1, y1]] ]
		"""
		points = []
		
		image = self.realsense.get_color_image_from_frames(frames)
		
		# green
		green_xy = self._get_green_control_xy(image, threshold)
		points.append(['green', green_xy])
		
		# TODO 换成老刘写的检测
		return points
	
	def _update_found_points(self, found_points) -> bool:
		"""
		添加匹配到的点坐标
		:param found_points: # [[点名: str, [Camera坐标X1, Y1, Z1], [Robot坐标X2, Y2, Z2]]]
		:return: 成功与否bool
		"""
		
		# [[点名: str, [Camera坐标x1, y1, z1], [Robot坐标x2, y2, z2]]]
		self._found_point_data.clear()
		
		def find_control(pname: str):
			for p in self.control_points:
				if p[0] == pname:
					return p
			return None
		
		for p in found_points:
			if isinstance(p[0], str) and 3 == p[1].__len__():
				control_point = find_control(p[0])
				self._found_point_data.append(
					[p[0], control_point[1], np.asarray(p[1])])
			else:
				Printer.print("%s error: " % sys._getframe().f_code.co_name, Printer.red)
				return False
		return True

	def _get_control_point_coor(self, point_name: str):
		"""
		从类中已加载的控制点列表返回坐标
		:param point_name: 点名
		:return: np.asarray([Robot坐标x2, y2, z2])
		"""
		for point in self.control_points:
			if point_name == point[0]:
				return point[1]
		return None
	
	"""处理目标点的函数"""
	# def _get_green_control(self, image, threshold, count_threshold = 30) -> np.array:
	# 	"""
	# 	获得绿色控制点位置,若找不到返回None
	# 	:param image: RGB图像
	# 	:param threshold: 可信度阈值
	# 	:return: 坐标[x, y]，若找不到返回None
	# 	"""
	# 	scale_percentage = 0.1
	# 	small_image = ImgProcessor.resize(image, scale_percentage)
	#
	# 	miu = np.asarray([0.230604265949190, 0.528880344417954, 0.477433844165268])
	# 	sig = np.asarray([[8.142333919138685e-05, -3.528365300776935e-04, 2.419821012445137e-05],
	# 	                  [-3.528365300776935e-04, 0.004714512541749, -0.001336533557575],
	# 	                  [2.419821012445137e-05, -0.001336533557575, 0.003038826607760]])
	#
	# 	green_binary = self._pick_color_mask(small_image, threshold, miu, sig)
	#
	# 	xy = ImgProcessor.get_mass_center(green_binary)
	# 	xy = np.round(xy / scale_percentage)
	# 	xy = np.asarray(xy, dtype=np.int)
	#
	# 	# TODO 判断如果没有的情况
	# 	hist = cv2.calcHist([green_binary], [0], None, [256], [0, 256])
	#
	# 	if not hist[255] > count_threshold:
	# 		xy = None
	#
	# 	return xy
	#
	# TODO green, blue, yellow
	
	def _get_color_center(self, image, lower, upper, count_threshold=30):

		scale_percentage = 0.1
		small_image = ImgProcessor.resize(image, scale_percentage)
		
		hsv_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
		hsv_image = np.asarray(hsv_image)
		
		mask = cv2.inRange(hsv_image, lower, upper)
		
		hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
		if not hist[255] > count_threshold:
			return None
		
		xy = ImgProcessor.get_mass_center_float(mask)
		xy = np.round(xy / scale_percentage)
		xy = np.asarray(xy, dtype=np.int)
		return xy
	
	def _get_green_control_xy(self, image):
		
		width = 10
		hue_c = 0.230604265949190 * 180
		lower = np.asarray([hue_c - width, 0.2 * 255, 0.35 * 255])
		upper = np.asarray([hue_c + width, 0.7 * 255, 0.75 * 255])
		
		xy = self._get_color_center(image, lower, upper)
		return xy  # TODO 写到这
	
	def test(self):
		image = cv2.imread('old_py/ControlPoint/green/001.png')
		xy = self._get_color_center(image)
		if xy is None:
			return
		print("green center:", xy)
		
		res_img = cv2.circle(image, tuple(xy), 3, (0, 255, 255))
		
		cv2.imshow('green center', res_img)
		cv2.waitKey(0)
	
	# @staticmethod
	# def _pick_color_mask(image, threshold, miu: np.array, sig: np.array):
	#
	# 	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	# 	hsv_image = np.asarray(hsv_image)
	#
	# 	mask = np.zeros([hsv_image.shape[0], hsv_image.shape[1], 1], dtype=np.uint8)
	#
	# 	signeg = np.linalg.inv(sig)
	# 	c = 1 / (2 * pi) ** 1.5 / np.linalg.det(sig) ** 0.5
	#
	# 	# 置信区间
	# 	large_miu = np.asarray([miu[0] * 180, miu[1] * 255, miu[2] * 255])
	#
	# 	# large_sigma = np.asarray([sig[0, 0] * 180, sig[1, 1] * 255, sig[2, 2] * 255]) * 3
	#
	# 	# 判断是否在 置信区间内
	# 	def if_in_confidence(x):
	# 		width = 10
	# 		if not large_miu[0] - width < x[0] < large_miu[0] + width:
	# 			return False
	# 		return True
	#
	# 	h, w, chanel = hsv_image.shape
	# 	for i in range(h):
	# 		for j in range(w):
	# 			xi = np.asarray(hsv_image[i][j])
	#
	# 			# 粗筛选
	# 			if not if_in_confidence(xi):
	# 				mask[i][j] = [0]
	# 				# print("skip")
	# 				continue
	#
	# 			xi = np.asarray([xi[0] / 180, xi[1] / 255, xi[2] / 255])
	# 			temp1 = np.transpose(xi - miu)
	# 			temp2 = np.matmul(temp1, signeg)
	# 			temp3 = np.matmul(temp2, (xi - miu))
	# 			upper = temp3 / -2
	#
	# 			pixi = c * exp(upper)
	#
	# 			if pixi > threshold:
	# 				mask[i][j] = [255]
	# 			else:
	# 				mask[i][j] = [0]
	# 	return mask
	#

	
	
if __name__ == '__main__':
	positioning = Positioning(RealsenseManager())
	positioning.load_control_point_file()

	positioning.test()
	
	
	# positioning.get_camera_position(positioning.realsense.get_aligned_frames(), 0.8)   # TODO
	# print('para:', positioning.trans_para)
	# print('origin', positioning.control_points[0][1])
	# print(positioning.coor_trans(positioning.control_points[0][1]))
	
	