# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import time
from typing import List
import glob
from math import sqrt

from RealsenseManager import *
from ConsolePrinter import Printer


class RealsenseCamera(RealsenseManager):
	"""
	第二版，项目中用的，带有坐标转换参数的相机管理器
	"""
	
	trans_para = np.zeros((6, 1), dtype=np.float)
	if_get_position_flag = False  # 是否成功解算相机位置
	
	_cp_coor_dic = {}  # 控制点坐标 { 点名str : [Robot坐标X， Y， Z] }
	_cp_image_dic = {}  # 控制点模板图片 { 点名str : image }
	
	# 匹配到的控制点坐标 {点名str : [Camera坐标x1, y1, z1] }
	_found_point_data_dic = {}
	
	def __init__(self, control_point_path='control_point_data/'):
		super().__init__()
		self.load_control_point_files(control_point_path)
		Printer.print('control point files loaded.', Printer.green)
	
	def get_transform_matrix(self, aligned_frames):
		self._found_point_data_dic.clear()
		
		# 检测目标点位置
		image = self.get_color_image_from_frames(aligned_frames)
		point_xy_dic = self._find_markers(image)
		
		# 转成相机空间坐标
		for pname, xy in point_xy_dic.items():
			XYZ = self.get_coor_in_Camera_system(aligned_frames, xy)
			if XYZ is not None:
				self._found_point_data_dic[pname] = XYZ
	
		# 解转换参数
		self.if_get_position_flag = self._solve_7_paras(sigma_threshold=0.1)
	
	def _solve_7_paras(self, sigma_threshold=0.1) -> bool:
		n_pt = self._found_point_data_dic.__len__()
		n_equ = n_pt * 3  # 方程数
		
		if n_equ < 6:  # 点数不够
			# Printer.print("Control point not enough. Can't solve.", Printer.yellow)
			return False
		
		B = np.zeros((n_equ, 6), dtype=np.float)
		L = np.zeros((n_equ, 1), dtype=np.float)
		
		# 每个点对应三个方程
		i = 0
		for pname, XYZc in self._found_point_data_dic.items():
			if pname not in self._cp_coor_dic.keys():
				continue
			
			xyz2 = self._cp_coor_dic[pname]
			x1, y1, z1 = XYZc
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
			if sigma > sigma_threshold:
				Printer.print("sigma > %f" % sigma_threshold, Printer.yellow)
				return False
			else:
				return True
		except Exception as e:
			Printer.print(e.__str__(), Printer.red)
			return False

	def coor_trans_positive(self, aligned_frames: rs.frame, xy: np.array):
		"""
		正向坐标变换，若转换失败返回None
		:param XYZ: 相机坐标系的坐标 XYZ
		:return: 返回转换后的np.array(XYZ2)， 若转换失败返回None
		"""
		if not self.if_get_position_flag:
			Printer.print("No camera position, Cannot trans coor.", Printer.red)
			return None
		
		XYZ = self.get_coor_in_Camera_system(aligned_frames, xy)
		
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
		XYZ2 = np.matmul(B, self.trans_para).reshape(3) + XYZ
		return XYZ2
	
	def load_control_point_files(self, path: str):
		self._cp_coor_dic.clear()
		self._cp_image_dic.clear()
		
		# 读坐标文件
		coor_filename = path + 'coors.txt'
		file = open(coor_filename)
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
				self._cp_coor_dic[pname] = np.asarray([X, Y, Z])
				
			except ValueError as e:
				Printer.print("Control point file ERROR." + e.__str__(), Printer.red)
		
		# 读模板图片
		img_name_ls = glob.glob(path + '*.jpg')
		for img_name in img_name_ls:
			self._cp_image_dic[img_name[:-4]] = cv2.imread(img_name)
		return
	
	def _find_markers(self, target_img) -> dict:
		"""
		找到所有标记
		:param target_img:
		:return: { 点名str : xy坐标np.array }
		"""
		display_img = target_img.copy()
		
		# Initiate SIFT detector创建sift检测器
		sift = cv2.xfeatures2d.SIFT_create()
		
		# compute the descriptors with ORB
		target_kp, target_des = sift.detectAndCompute(target_img, None)
		
		result_xy_dic = {}
		
		for imgname, template_img in self._cp_image_dic.items():
			marker_center = self._match_template_SIFT(template_img, target_kp, target_des, 0.8,
			                                       draw_img=display_img, min_match_count=20)
			if marker_center is not None:
				result_xy_dic[imgname] = marker_center
		
		return result_xy_dic
		
	def _match_template_SIFT(self, template_img, target_kp, target_des, threshold, draw_img=None, min_match_count=25):
		"""
		用SIFT进行模板匹配，如果找到了返回center，如果没找到返回None
		:param template_img:
		:param target_kp:
		:param target_des:
		:param threshold:
		:param draw_img:
		:param min_match_count:
		:return:
		"""
		# Initiate SIFT detector创建sift检测器
		sift = cv2.xfeatures2d.SIFT_create()
		
		# compute the descriptors with ORB
		template_kp, template_des = sift.detectAndCompute(template_img, None)
		
		# 创建设置FLANN匹配
		FLANN_INDEX_KDTREE = 0
		# 匹配算法
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		# 搜索次数
		search_params = dict(checks=30)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		
		matches = flann.knnMatch(template_des, target_des, k=2)
		
		# store all the good matches as per Lowe's ratio test.
		good_match = []
		# 舍弃大于threshold的匹配, threshold越小越严格
		for m, n in matches:
			if m.distance < threshold * n.distance:
				good_match.append(m)
		print("matches found - %d/%d" % (len(good_match), min_match_count))
		
		if len(good_match) > min_match_count:
			# 获取关键点的坐标
			src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
			dst_pts = np.float32([target_kp[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
			# 计算变换矩阵和MASK
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			# matchesMask = mask.ravel().tolist()
			h, w = template_img.shape[:2]
			# 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
			pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
			dst = cv2.perspectiveTransform(pts, M)
			center = np.asarray([np.mean(dst[:, 0, 0]), np.mean(dst[:, 0, 1])])

			cv2.polylines(draw_img, [np.int32(dst)], True, [0, 0, 255], 2, cv2.LINE_AA)
			cv2.circle(draw_img, (center[0], center[1]), 3, (0, 255, 0), 3)
		else:
			center = None
		
		return center
	
	def get_found_point_dic(self) -> dict:
		return self._found_point_data_dic


if __name__ == '__main__':
	realsense = RealsenseCamera()
	aligned_frames = realsense.get_aligned_frames()
	realsense.get_transform_matrix(aligned_frames)
	
	# 测试
	found_point_dic = realsense.get_found_point_dic()
	xy = found_point_dic['sign1']
	XYZ = realsense.coor_trans_positive(aligned_frames, xy)
	print(XYZ)
	
	
	

		
	
			
			
			

