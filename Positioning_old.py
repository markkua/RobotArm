# -*- encoding: utf-8 -*-

from cv2 import *
from typing import List
import os
import numpy as np
from math import sin, cos, pi


class Positioning:
	
	Xc, Yc, Zc = [0.0] * 3
	phi, omega, kappa = [0.0] * 3
	
	sign_template_dir = "positioning_data"
	
	def get_camera_position(self, frameset):
		self._find_signs(frameset)
		pass
	
	def _find_signs(self, frameset) -> List:
		workdir = os.getcwd()
		sign_dir = workdir + "/" + self.sign_template_dir + "/"
		
		pass
	
	def _match_sign(self, img, sign_img, hi) -> (float, float):
		"""
		匹配单个标记的RGB位置
		:param img: 摄像机影像
		:param sign_img: 预设的标志影像
		:return: 标记的坐标
		"""
		# TODO SIFT 匹配
		pass
	
	def _solve_paras(self):
		"""
		least_squares_method
		:return:
		"""
		pass
		
	def test_R(self, angle_arr, xyz_arr):
		""" 测试旋转矩阵 """
		print("init=\n", xyz_arr)
		Rx, Ry, Rz = calculate_R(angle_arr)
		R = np.matmul(np.matmul(Rx, Ry), Rz)
		print("R=\n", R)
		temp = np.matmul(R, xyz_arr)
		print("temp=\n", temp)
		Rx, Ry, Rz = calculate_R(-1 * angle_arr)
		R = np.matmul(np.matmul(Rz, Ry), Rx)
		print("R2=\n", R)
		result = np.matmul(R, temp)
		print("result=\n", result)


"""
	基础公用函数
"""


def calculate_R(angel_arr):
	print("angle_arr=", angel_arr)
	phi, omega, kappa = angel_arr[:3]
	Rz = np.asarray([
		[cos(kappa), -sin(kappa), 0],
		[sin(kappa), cos(kappa), 0],
		[0, 0, 1]
	])
	Rx = np.asarray([
		[1, 0, 0],
		[0, cos(omega), -sin(omega)],
		[0, sin(omega), cos(omega)]
	])
	Ry = np.asarray([
		[cos(phi), 0, sin(phi)],
		[0, 1, 0],
		[-sin(phi), 0, cos(phi)]
	])
	return Rx, Ry, Rz


def deg2rad(deg: float):
	return deg * pi / 180


def rad2deg(rad: float):
	return rad * 180 / pi

		
if __name__ == '__main__':
	positioning = Positioning()
	positioning.test_R(np.asarray([0, 20, 0]), [2, 1, 5])
