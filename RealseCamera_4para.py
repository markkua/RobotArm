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
	
	trans_para_7 = np.zeros((6, 1), dtype=np.float)
	trans_para_4 = np.zeros((4, 1), dtype=np.float)
	
	if_get_position_flag = False  # 是否成功解算相机位置
	
	_cp_coor_dic = {}  # 控制点坐标 { 点名str : [Robot坐标X， Y， Z] }
	_cp_image_dic = {}  # 控制点模板图片 { 点名str : image }
	
	# 匹配到的控制点坐标 {点名str : [Camera坐标x1, y1, z1] }
	_found_point_XYZc_dic = {}
	
	perspective_matrix = np.array((3, 3), dtype=np.float32)
	
	def __init__(self, control_point_path='control_point_data/'):
		super().__init__()
		self.load_control_point_files(control_point_path)
		Printer.print('control point files loaded.', Printer.green)
	
	def get_transform_matrix(self, aligned_frames, sigma_threshold=10) -> np.array:
		self._found_point_XYZc_dic.clear()
		
		# 检测目标点位置
		image = self.get_color_image_from_frames(aligned_frames)
		
		point_xy_dic, display_img = self._find_markers(image)
		
		self.point_xy_dic = point_xy_dic  # TODO 测试用
		
		# 转成相机空间坐标
		for pname, xy in point_xy_dic.items():
			XYZ = self.pixelxy2cameraXYZ(aligned_frames, xy)
			if XYZ is not None:
				self._found_point_XYZc_dic[pname] = XYZ
	
		# 解转换参数
		# if self._solve_7_paras(sigma_threshold=sigma_threshold):
		# if self._solve_perspect_M():
		if self._solve_para_4():
			self.if_get_position_flag = True
			Printer.print('transform matrix get.', Printer.green)
		else:
			self.if_get_position_flag = False
		return display_img
		
	def _solve_7_paras(self, sigma_threshold) -> bool:
		n_pt = self._found_point_XYZc_dic.__len__()
		n_equ = n_pt * 3  # 方程数

		if n_equ < 6:  # 点数不够
			# Printer.print("Control point not enough. Can't solve.", Printer.yellow)
			return False

		B = np.zeros((n_equ, 6), dtype=np.float)
		L = np.zeros((n_equ, 1), dtype=np.float)

		# 每个点对应三个方程
		i = 0
		for pname, XYZ1 in self._found_point_XYZc_dic.items():
			xyz2 = self._cp_coor_dic[pname]
			x1, y1, z1 = XYZ1
			x2, y2, z2 = xyz2

			B[3 * i, :] = [1, 0, 0, 0, -z1, y1]
			B[3 * i + 1, :] = [0, 1, 0, z1, 0, -x1]
			B[3 * i + 2, :] = [0, 0, 1, -y1, x1, 0]

			L[3 * i:3 * i + 3, 0] = [x2 - x1, y2 - y1, z2 - z1]

			i += 1

		# 解方程
		try:
			# B = B * -1
			NBB = np.matmul(B.transpose(), B)
			NBB_inv = np.linalg.inv(NBB)
			BTPL = np.matmul(B.transpose(), L)
			result = np.matmul(NBB_inv, BTPL)
			self.trans_para_7[:, 0] = result[:, 0]

			V = np.matmul(B, result) - L
			r = n_equ - 6
			VTV = np.matmul(V.transpose(), V)
			sigma = sqrt(VTV[0, 0] / r)
			print('sigma=', sigma)
			if sigma > sigma_threshold:
				Printer.print("sigma > %f" % sigma_threshold, Printer.yellow)
				return False
			else:
				return True
		except Exception as e:
			Printer.print('matrix error: ' + e.__str__(), Printer.red)
			return False
	
	def _solve_para_4(self):
		n_equ = len(self.point_xy_dic)
		if n_equ < 4:
			return False
	
		B = np.zeros((n_equ, 4), dtype=np.float)
		L = np.zeros((n_equ, 1), dtype=np.float)
	
		i = 0
		for pname, xy in self.point_xy_dic.items():
			x, y = xy
			x2, y2, z2 = self._cp_coor_dic[pname]
			B[2*i, :] = [1, 0, x, y]
			B[2*i+1, :] = [0, 1, -x, y]
			
			L[2*i, 0] = x2 - x
			L[2*i+1, 0] = y2 - y
		
		try:
			NBB = np.matmul(B.transpose(), B)
			NBB_inv = np.linalg.inv(NBB)
			BTPL = np.matmul(B.transpose(), L)
			result = np.matmul(NBB_inv, BTPL)
			self.trans_para_4[:, 0] = result[:, 0]

			# V = np.matmul(B, result) - L
			# r = n_equ - 4
			# VTV = np.matmul(V.transpose(), V)
			# sigma = sqrt(VTV[0, 0] / r)
			
			return True
		except Exception as e:
			return False
	
	def _solve_perspect_M(self) -> bool:
		if self.point_xy_dic.__len__() < 4:
			return False
		
		src_points = []
		dst_points = []
		for pname, xy in self.point_xy_dic.items():
			src_points.append(xy)
			dst_points.append(self._cp_coor_dic[pname][0:2])
		
		try:
			M = cv2.getPerspectiveTransform(src_points, dst_points)
			self.perspective_matrix = M
			return True
		except Exception as e:
			Printer.print('perspect M error: ', e.__str__, Printer.red)
			return False
		
	def coor_trans_pixelxy2worldXYZ(self, aligned_frames: rs.frame, xy: np.array):
		if not self.if_get_position_flag:
			# Printer.print("No camera position, Cannot trans coor.", Printer.red)
			return None
		
		if len(xy) != 2:
			Printer.print('coor trans len(xy) != 2', Printer.red)
			return None
		
		# 七参数
		XYZ_c = self.pixelxy2cameraXYZ(aligned_frames, xy)
		if XYZ_c is None:
			return None
		#
		XYZ_c = np.asarray(XYZ_c)
		if 3 != XYZ_c.size:
			Printer.print("coor trans error: xyz size wrong.", Printer.red)
			return None
		XYZ_r = self._coor_trans_cameraXYZ2worldXYZ(XYZ_c)
		return XYZ_r
	
		# 投影变换方法
		# pts = np.asarray([xy[0], xy[1], 1])
		# dst = np.matmul(pts, self.perspective_matrix)
		#
		# XYZ = np.asarray([dst[0], dst[1], 0])
		
	def _coor_trans_cameraXYZ2worldXYZ(self, XYZ):
		"""
		正向坐标变换，若转换失败返回None
		:param XYZ: 相机坐标系的坐标 XYZ
		:return: 返回转换后的np.array(XYZ2)， 若转换失败返回None
		"""
		if not self.if_get_position_flag:
			# Printer.print("No camera position, Cannot trans coor.", Printer.red)
			return None
		
		# 七参数
		# x1, y1, z1 = XYZ
		# B = np.asarray([
		# 	[1, 0, 0, 0, -z1, y1],
		# 	[0, 1, 0, z1, 0, -x1],
		# 	[0, 0, 1, -y1, x1, 0]
		# ])
		# XYZ2 = np.matmul(B, self.trans_para_7).reshape(3) + XYZ
		
		# 四参数
		x1, y1, z1 = XYZ
		B = np.asarray([
		
		])
		
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
			self._cp_image_dic[img_name[len(path):-4]] = cv2.imread(img_name)
		return
	
	def _find_markers(self, target_img):
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
		
		return result_xy_dic, display_img
		
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
		
		if len(good_match) > min_match_count:
			Printer.print("matches found - %d/%d" % (len(good_match), min_match_count), Printer.green)
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
			center = np.asarray([np.round(np.mean(dst[:, 0, 0])), np.round(np.mean(dst[:, 0, 1]))], dtype=np.int)

			cv2.polylines(draw_img, [np.int32(dst)], True, [0, 0, 255], 2, cv2.LINE_AA)
			cv2.circle(draw_img, (center[0], center[1]), 3, (0, 255, 0), 3)
		else:
			Printer.print("matches found - %d/%d" % (len(good_match), min_match_count), Printer.yellow)
			center = None
		
		return center
	
	def get_found_point_dic(self) -> dict:
		return self._found_point_XYZc_dic
	
	def test_solve(self):
		self._found_point_XYZc_dic.clear()
		self._found_point_XYZc_dic = {
			'1': [-2085738.7757, 5503702.8697, 2892977.6829],
			'2': [-2071267.5135, 5520926.7235, 2883341.8135],
			# '3': [-2079412.5535, 5512450.8800, 2879771.2119],
			'4': [-2093693.1744, 5511218.2651, 2869861.8947],
			'5': [-2113681.5062, 5491864.0382, 2896934.4852],
			'6': [-2100573.2849, 5496675.0138, 2894377.6030]
		}
		
		self._cp_coor_dic.clear()
		self._cp_coor_dic = {
			'1': np.asarray([-2085635.1879, 5503757.4154, 2892982.0896]),
			'2': np.asarray([-2071164.1636, 5520981.4653, 2883346.1670]),
			'3': [-2079308.9840, 5512505.3689, 2879775.4919],
			'4': np.asarray([-2093589.3723, 5511272.3144, 2869866.0221]),
			'5': [-2113577.7476, 5491917.9895, 2896938.5457],
			'6': [-2100469.5957, 5496729.2165, 2894381.7872]
		}
		
		self.if_get_position_flag = self._solve_7_paras(sigma_threshold=100)
		
		print('para:', self.trans_para_7)
		
		XYZ = self._coor_trans_cameraXYZ2worldXYZ(self._found_point_XYZc_dic['1'])
		# XYZ = self.coor_trans_cameraXYZ2worldXYZ([0, 0, 0])
		
		print(XYZ)


if __name__ == '__main__':
	realsense = RealsenseCamera()
	# realsense.test_solve()
	while True:
		aligned_frames = realsense.get_aligned_frames()
		image = realsense.get_color_image_from_frames(aligned_frames)

		# 测试
		start_time = time.time()
		display_img = realsense.get_transform_matrix(aligned_frames, sigma_threshold=100)

		cv2.imshow('test', cv2.resize(display_img, (int(1920/2), int(1080/2))))
		cv2.waitKey(1)

		# if realsense.if_get_position_flag:
		# 	perspectiveImg = cv2.warpPerspective(image, realsense.perspective_matrix, (400, 600))
		# 	cv2.imshow('perspective', perspectiveImg)

		found_point_dic = realsense.get_found_point_dic()
		print('time=', time.time() - start_time)

		print('found point:', found_point_dic)

		# if 'sign2' in found_point_dic.keys():
		# 	xyz2 = found_point_dic['sign2']
		# 	xy_pixel = realsense.point_xy_dic['sign2']
		# 	print('sign2 xyz=', xyz2)
		# 	# XYZ = realsense.coor_trans_pixelxy2worldXYZ(aligned_frames, xy_pixel)
		# 	XYZ = realsense.coor_trans_cameraXYZ2worldXYZ(xyz2)
		# 	print('sign2 XYZ=', XYZ)
		#
		# if 'sign3' in found_point_dic.keys():
		# 	xyz3 = found_point_dic['sign3']
		# 	xy_pixel = realsense.point_xy_dic['sign3']
		# 	print('sign3 xyz=', xyz3)
		# 	XYZ = realsense.coor_trans_cameraXYZ2worldXYZ(xyz3)
		# 	print('sign3 XYZ=', XYZ)
		# if found_point_dic.__len__() == 3:
		# 	print(found_point_dic)
		# 	pass
		#
		if 'sign5' in found_point_dic.keys() and 'sign6' in found_point_dic.keys():
			xy5 = realsense.point_xy_dic['sign5']
			xy6 = realsense.point_xy_dic['sign6']
			
			xyz5 = found_point_dic['sign5']
			xyz6 = found_point_dic['sign6']

			# xyz5[0] *= -1

			XYZ5 = realsense.coor_trans_pixelxy2worldXYZ(aligned_frames, xy5)
			print('xy5=', xy5)
			print('xyz5=', xyz5)
			print('XYZ5=', XYZ5)
			
			XYZ6 = realsense.coor_trans_pixelxy2worldXYZ(aligned_frames, xy6)
			print('xy6=', xy6)
			print('xyz6=', xyz6)
			print('XYZ6=', XYZ6)
			
		#
		#
		# 	#
		# 	# delta = np.asarray(xyz3) - np.asarray(xyz2)
		# 	# print('delta=', delta)
		# 	# print('distance=', sum(delta**2)**0.5)
		#
		# key = cv2.waitKey(1)
		# if 27 == key:
		# 	break



	
			
			
			
	
