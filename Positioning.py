# -*- encoding: utf-8 -*-

from RealsenseManager import RealsenseManager

from cv2 import *
from typing import List
import os
import numpy as np
from math import sin, cos, pi


class Positioning:
	
	# 坐标转换参数
	class TransPara_6:
		# 平移参数
		Dx: float = 0.0
		Dy: float = 0.0
		Dz: float = 0.0
		# 旋转参数
		Rx: float = 0.0
		Ry: float = 0.0
		Rz: float = 0.0
	
	trans_para = TransPara_6()
	
	_point_data = []  # 控制点坐标 [[Camera坐标x, y, z], [Robot坐标x, y, z]]
	
	
	def get_camera_position(self, frames):
		pass
	
	def _solve_7_paras(self, point_ls):
		if len(point_ls) * 3 < 7:
			return False
		

if __name__ == '__main__':
	pass