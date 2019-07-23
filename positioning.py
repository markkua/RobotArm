# -*- encoding: utf-8 -*-

from cv2 import *
from typing import List


class Positioning:
	
	Xc, Yc, Zc = [0.0] * 3
	phi, omega, kappa = [0.0] * 3
	
	def get_camera_position(self, frameset):
		
		pass
	
	def _find_signs(self) -> List:
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
	
	def test(self):
		print(self.Xc, self.Zc)
		
		
if __name__ == '__main__':
	positioning = Positioning()
	positioning.test()
