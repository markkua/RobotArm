# -*- encoding: utf-8 -*-


from ConsolePrinter import Printer
from RealsenseManager import RealsenseManager

import pyrealsense2 as rs
import numpy as np
import cv2
from typing import List
import os
import sys
from math import sin, cos, pi, sqrt, exp
import time


class RealsenseCamera(RealsenseManager):
	"""
	项目中用的，带有坐标转换参数的相机管理器
	"""
	
	coor_filename = "data/control_points.txt"
	
	trans_para = np.zeros((6, 1), dtype=np.float)
	
	control_points = []  # [ [点名:str, [Robot坐标x2, y2, z2]] ]
	
	# 匹配到的控制点坐标 [ [点名:str, [Camera坐标x1, y1, z1], [Robot坐标x2, y2, z2]] ]
	_found_point_data = []
	
	mask_img = None
	
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
	
	def get_transform_matrix(self, aligned_frames, hue_width=10):
		# 检测目标点位置
		image = self.get_color_image_from_frames(aligned_frames)
		points_xy_ls = self._match_control_points(image, hue_width)
		
		# 转成相机空间坐标
		points_XYZ_ls = []
		for xy_point in points_xy_ls:
			print('xy_point=', xy_point)  # TODO
			XYZ = self.get_coor_in_Camera_system(aligned_frames, [xy_point[1][0], xy_point[1][1]])  # 这个函数只能传入list， 不能是array
			if XYZ is not None:
				points_XYZ_ls.append([xy_point[0], XYZ])

		# update_found_points
		self._update_found_points(points_XYZ_ls)
		
		# 解参数
		self.if_get_position = self._solve_7_paras()
	
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
			# Printer.print("Control point not enough. Can't solve.", Printer.yellow)
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
			
			L[3 * i:3 * i + 3, 0] = [x2 - x1, y2 - y1, z2 - z1]
		
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
	
	def _match_control_points(self, image, hue_width: int = 10, count_threshold=30, hue_bias=0) -> List:
		"""
		匹配目标点，返回[ [点名:str, [图像坐标x1, y1]] ]
		:param frames:
		:return:  [ [点名:str, [图像坐标x1, y1]] ]
		"""
		points_xy_ls = []
		
		hue_c_green = 0.230604265949190 * 180
		hue_c_red = 0.957300173750032 * 180
		hsv_range_ls = [
			['green',
			 [
				 np.asarray([hue_c_green - hue_width + hue_bias, 0.2 * 255, 0.35 * 255]),
				 np.asarray([hue_c_green + hue_width + hue_bias, 0.7 * 255, 0.75 * 255])
			 ]],
			['red',
			 [
				 np.asarray([hue_c_red - hue_width*0.8 + hue_bias, 0.5 * 255, 0.2 * 255]),
				 np.asarray([180, 0.9 * 255, 0.8 * 255])
			 ]]
		]
		
		mask_ls = []
		for hsv_range in hsv_range_ls:
			xy, mask = self._get_color_center_xy(image, hsv_range[1][0], hsv_range[1][1], count_threshold)
			if xy is not None:
				mask_ls.append(mask)
				self.mask_img = mask
				points_xy_ls.append([hsv_range[0], xy])
		# self.mask_img = cv2.bitwise_or(mask_ls[0], mask_ls[1])  # TODO 多个mask

		return points_xy_ls
	
	def _get_color_center_xy(self, image, lower, upper, count_threshold):
		scale_percentage = 0.1
		small_image = self._resize(image, scale_percentage)
		
		hsv_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
		hsv_image = np.asarray(hsv_image)
		
		mask = cv2.inRange(hsv_image, lower, upper)
		
		# cv2.imshow('mask', mask)
		
		hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
		if not hist[255] > count_threshold:
			return None, None
		
		xy = self._get_mass_center_float(mask)
		xy = np.round(xy / scale_percentage)
		xy = np.asarray(xy, dtype=np.int)
		return xy, mask

	def _update_found_points(self, camera_nameXYZ_ls) -> bool:
		"""
		添加匹配到的点坐标
		:param camera_nameXYZ_ls: # [[点名: str, [Camera坐标X1, Y1, Z1]]]
		:return: 成功与否bool
		"""

		self._found_point_data.clear()

		def find_control(pname: str):
			for p in self.control_points:
				if p[0] == pname:
					return p
			return None

		for p in camera_nameXYZ_ls:
			if isinstance(p[0], str) and 3 == p[1].__len__():
				control_point = find_control(p[0])
				self._found_point_data.append(
					[p[0], control_point[1], np.asarray(p[1])])
			else:
				Printer.print("%s error: " % sys._getframe().f_code.co_name, Printer.red)
				return False
		return True
	#
	# def _get_green_control_xy(self, image, hue_width=10, count_threshold=30):
	# 	scale_percentage = 0.1
	# 	small_image = self._resize(image, scale_percentage)
	#
	# 	hue_c = 0.230604265949190 * 180
	# 	lower = np.asarray([hue_c - hue_width, 0.2 * 255, 0.35 * 255])
	# 	upper = np.asarray([hue_c + hue_width, 0.7 * 255, 0.75 * 255])
	#
	# 	hsv_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
	# 	hsv_image = np.asarray(hsv_image)
	#
	# 	mask = cv2.inRange(hsv_image, lower, upper)
	#
	# 	hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
	# 	if not hist[255] > count_threshold:
	# 		return None
	#
	# 	xy = self._get_mass_center_float(mask)
	# 	xy = np.round(xy / scale_percentage)
	# 	xy = np.asarray(xy, dtype=np.int)
	# 	return xy
	#
	# def _get_red_control_xy(self, image, hue_width=10, count_threshold=30):
	# 	# 红色分两块，拼接组合
	# 	scale_percentage = 0.1
	# 	small_image = self._resize(image, scale_percentage)
	#
	# 	hue_c = 0.957300173750032 * 180
	# 	lower_1 = np.asarray([hue_c - hue_width*0.8, 0.5 * 255, 0.2 * 255])
	# 	upper_1 = np.asarray([180, 0.9 * 255, 0.8 * 255])
	# 	# lower_2 = np.asarray([0, 0.5*255, 0.2 * 255])
	# 	# upper_2 = np.asarray([0.5*180, 0.9 * 255, 0.8 * 255])
	#
	# 	hsv_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2HSV)
	# 	hsv_image = np.asarray(hsv_image)
	#
	# 	mask1 = cv2.inRange(hsv_image, lower_1, upper_1)
	# 	# mask2 = cv2.inRange(hsv_image, lower_2, upper_2)
	# 	# 或运算
	# 	# mask = cv2.bitwise_or(mask1, mask2)
	# 	mask = mask1
	#
	# 	hist = cv2.calcHist([mask], [0], None, [256], [0, 256])
	# 	if not hist[255] > count_threshold:
	# 		return None
	#
	# 	xy = self._get_mass_center_float(mask)
	# 	xy = np.round(xy / scale_percentage)
	# 	xy = np.asarray(xy, dtype=np.int)
	# 	return xy
	#
	
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

	@staticmethod
	def _resize(image, percentage):
		h, w, c = image.shape
		return cv2.resize(image, (int(w * percentage), int(h * percentage)))
	
	@staticmethod
	def _get_mass_center_float(gray_img):
		result = cv2.moments(gray_img, 1)
		if not 0 == result['m00']:
			x = round(result['m10'] / result['m00'])
			y = round(result['m01'] / result['m00'])
		else:
			x, y = 0, 0
		return np.asarray([x, y])
	
	def test(self, image):
		while True:
			frame = self.get_aligned_frames()
			image = self.get_color_image_from_frames(frame)

			xy_ls = self._match_control_points(image, 10)
			print('xy_ls: ', xy_ls)
			
			res_img = cv2.circle(image, tuple(xy_ls[0][1]), 3, (0, 255, 255))
			
			return res_img
			
	# end of class


# def get_color_center(image, lower, upper):
# 	hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# 	hsv_image = np.asarray(hsv_image)
#
# 	mask = cv2.inRange(hsv_image, lower, upper)
#
# 	return mask

	



if __name__ == '__main__':
	camera = RealsenseCamera()
	camera.test()