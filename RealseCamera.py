# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import time
from typing import List

from RealsenseManager import *
from ConsolePrinter import Printer


class RealsenseCamera(RealsenseManager):
	"""
	第二版，项目中用的，带有坐标转换参数的相机管理器
	"""
	
	trans_para = np.zeros((6, 1), dtype=np.float)
	if_get_position = False  # 是否成功解算相机位置
	
	_control_points = {}  # { 点名str : [Robot坐标X， Y， Z] }
	
	# 匹配到的控制点坐标 {点名str:[[Camera坐标x1, y1, z1], [Robot坐标x2, y2, z2]]}
	_found_point_data = {}
	
	def __init__(self, control_p_filename='data/control_points.txt'):
		super().__init__()
		self._load_control_point_file(control_p_filename)
	
	def solve_transform_matrix(self, aligned_frames, hue_bias=0):
		pass
	
	def coor_trans_positive(self, XYZ: np.array):
		"""
		正向坐标变换，若转换失败返回None
		:param XYZ: 相机坐标系的坐标 XYZ
		:return: 返回转换后的np.array(XYZ2)， 若转换失败返回None
		"""
		if not self.if_get_position:
			Printer.print("No camera position, Cannot trans coor.", Printer.red)
			return None
		
		XYZ = np.asarray(XYZ)
		if 3 != XYZ.size:
			Printer.print("coor trans error: xyz size wrong.", Printer.red)
			return None
		
		x1, y1, z1 = XYZ
		B = np.asarray([
			[1, 0, 0, 0, -z1, y1],
			[0, 1, 0, z1, 0, -x1],
			[0, 0, 1, -y1, x1, 0]
		])
		xyz2 = np.matmul(B, self.trans_para).reshape(3) + XYZ
		return xyz2
	
	
	def _load_control_point_file(self, filename):
		self._control_points.clear()
		file = open(filename)
		content_ls = file.readlines()
		
		for line in content_ls:
			if '' == line:  # 空行
				continue
			line.replace(' ', '')  # 空格
			
			try:
				pname, X, Y, Z = line.split(',')
				X = float(X)
				Y = float(Y)
				Z = float(Z)
				self._control_points[pname] = np.asarray([X, Y, Z])
				
			except ValueError as e:
				Printer.print("Control point file ERROR." + e.__str__(), Printer.red)
		
	def _find_markers(self):
		pass
	
		
if __name__ == '__main__':
	realsense = RealsenseCamera()
	# realsense.test_circle()
	

		
	
			
			
			

