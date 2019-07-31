# -*- encoding: utf-8 -*-

from RealsenseManager import RealsenseManager
from ConsolePrinter import Printer

from cv2 import *
from typing import List
import os
import sys
import numpy as np
from math import sin, cos, pi, sqrt


class Positioning:
	
	coor_filename = "data/control_points.txt"
	
	trans_para = np.zeros((6, 1), dtype=np.float)
	if_get_position = False  # 是否成功解算相机位置
	
	control_points = []  # [ [点名:str, [Robot坐标x2, y2, z2]] ]
	
	# 匹配到的控制点坐标 [ [点名:str, [Camera坐标x1, y1, z1], [Robot坐标x2, y2, z2]] ]
	_found_point_data = []
	
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
		
	def get_camera_position(self, aligned_frames):
		# 检测目标点位置
		points = self._match_control_points(aligned_frames)
		if not self._update_found_points(points):
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
		
	def coor_trans_reverse(self, xyz2: np.array):
		"""坐标逆变换"""
		pass
	
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

	def _get_control_point_coor(self, point_name: str):
		for point in self.control_points:
			if point_name == point[0]:
				return point[1]
		return None
	
	def _match_control_points(self, frames):
		points = self._test_match_points()

		# TODO 老刘写的检测
		return points
	
	def _update_found_points(self, found_points) -> bool:
		"""添加匹配到的点坐标"""
		
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
	
	def _test_match_points(self):
		points = [
			['1',[-2085738.7757,5503702.8697,2892977.6829]],
			['2',[-2071267.5135,5520926.7235,2883341.8135]],
			['3',[-2079412.5535,5512450.8800,2879771.2119]],
			['4',[-2093693.1744,5511218.2651,2869861.8947]],
			['5', [-2113681.5062, 5491864.0382, 2896934.4852]],
			['6', [-2100573.2849, 5496675.0138, 2894377.6030]]
		]
		return points

if __name__ == '__main__':
	positioning = Positioning()
	positioning.load_control_point_file()
	positioning.get_camera_position(123)   # TODO
	print('para:', positioning.trans_para)
	print('origin', positioning.control_points[0][1])
	print(positioning.coor_trans(positioning.control_points[0][1]))
	
	