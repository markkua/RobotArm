# -*- encoding: utf-8 -*-

# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
import time
import os
from typing import List
import pyrealsense2 as rs


class TemplateMatcher:
	# SIFT parameter
	MIN_MATCH_COUNT = 15  # 设置最低特征点匹配数量为10
	select_threshold = 0.9
	
	# 公共变量
	tpl_path = ""  # template_path
	tpl_files = []  # template_files
	
	def __init__(self, MIN_MATCH_COUNT=15, select_threshold=0.9):
		# init data path
		cwd = os.getcwd()
		self.set_template_dir(cwd + '\\positioning_data\\')
		# init SIFT
		self.MIN_MATCH_COUNT = MIN_MATCH_COUNT
		self.select_threshold = select_threshold
	
	def set_template_dir(self, data_dir: str):
		self.tpl_path = data_dir
		self.tpl_files.clear()
		for file in os.listdir(self.tpl_path):
			if file.lower().endswith('.jpg'):
				self.tpl_files.append(file)
		
	def solve_camera_pose(self, aligned_frames):
		# Get aligned frames
		color_frame = aligned_frames.get_color_frame()
		
		# Validate that frame is valid
		if not color_frame:
			print("Color frame not valid!")
			return None
		
		color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
		
		output_img, target_list = self._match_templates(color_image)
		# TODO 返回target_list
		return output_img
		
	def _match_templates(self, image) -> [List, List]:
		"""
		在数据路径中，逐个进行模板匹配，并返回对应的像素坐标及可信度[point_name, x, y, %]
		:param image: 要进行识别的图像数据
		:return: image, [point_name, x, y, %]
		"""
		# 判断路径是否为空
		if 0 == self.tpl_files.__len__():
			print("Path is empty.")
			return []

		# record start time
		start_time = time.time()
		print("Start matching. ", start_time)
		
		# Initiate SIFT detector创建sift检测器
		sift = cv2.xfeatures2d.SIFT_create()
		
		# find the keypoints and descriptors of color_image with SIFT
		key_point_color, des_color = sift.detectAndCompute(image, None)
		
		output_img = image.copy()
		for file in self.tpl_files:
			filename = self.tpl_path + file
			template_img = cv2.imread(filename, cv2.IMREAD_COLOR)
			
			# SIFT
			key_point, des = sift.detectAndCompute(template_img, None)
			print("Find %d key points in %s" % (key_point.__len__(), file))
			
			# FLANN
			output_img = self._FLANN_match(key_point_color, des_color, key_point, des, template_img, output_img)
			# TODO 处理匹配点
		return output_img, []		# TODO 返回[point_num, x, y, %]
	
	def _FLANN_match(self, key_point1, des1, key_point2, des2, template_img, target_img):
		# 创建设置FLANN匹配
		FLANN_INDEX_KDTREE = 0
		# 匹配算法
		index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
		# 搜索次数
		search_params = dict(checks=10)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1, des2, k=2)
		print("Find %d FLANN matches" % matches.__len__())
		
		# store all the good matches as per Lowe's ratio test.
		good_match = []
		
		# 舍弃大于threshold的匹配, threshold越小越严格
		for m, n in matches:
			if m.distance < self.select_threshold * n.distance:
				good_match.append(m)
		print("Find %d good match points with threshold=%.3f" % (good_match.__len__(), self.select_threshold))
		
		if len(good_match) > self.MIN_MATCH_COUNT:
			# 获取关键点的坐标
			src_pts = np.float32([key_point1[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
			dst_pts = np.float32([key_point2[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
			# 计算变换矩阵和MASK
			M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
			matchesMask = mask.ravel().tolist()
			h, w = template_img.shape[:2]
			# 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
			pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
			dst = cv2.perspectiveTransform(pts, M)
			
			# 画出识别到的区域
			cv2.polylines(target_img, [np.int32(dst)], True, [0, 0, 255], 2, cv2.LINE_AA)
			# TODO 找中心点
		else:
			print("Not enough matches are found - %d/%d" % (len(good_match), self.MIN_MATCH_COUNT))
			matchesMask = None
		
		draw_params = dict(matchColor=(255, 0, 0),
		                   # singlePointColor=(0, 0, 255),
		                   matchesMask=matchesMask,
		                   flags=2)
		target_img = cv2.drawMatches(template_img, key_point1, target_img, key_point2, good_match, None, **draw_params)
		
		# TODO 画匹配点
		
		return target_img  # TODO 坐标


def test(matcher):
	# Create a pipeline
	pipeline = rs.pipeline()
	
	# Create a config and configure the pipeline to stream
	#  different resolutions of color and depth streams
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
	
	# Start streaming
	profile = pipeline.start(config)
	
	# Getting the depth sensor's depth scale (see rs-align example for explanation)
	depth_sensor = profile.get_device().first_depth_sensor()
	depth_scale = depth_sensor.get_depth_scale()
	print("Depth Scale is: ", depth_scale)
	
	# We will be removing the background of objects more than
	#  clipping_distance_in_meters meters away
	clipping_distance_in_meters = 1  # 1 meter
	clipping_distance = clipping_distance_in_meters / depth_scale
	
	# Create an align object
	# rs.align allows us to perform alignment of depth frames to others frames
	# The "align_to" is the stream type to which we plan to align depth frames.
	align_to = rs.stream.color
	align = rs.align(align_to)
	
	# Streaming loop
	try:
		while True:
			# Get frameset of color and depth
			frames = pipeline.wait_for_frames()
			# frames.get_depth_frame() is a 640x360 depth image
			
			# Align the depth frame to color frame
			aligned_frames = align.process(frames)
			
			images = matcher.solve_camera_pose(aligned_frames)
			
			# Render images
			cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Align Example', images)
			key = cv2.waitKey(1)  # 延迟
			# Press esc or 'q' to close the image window
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				break
			elif key == ord('s'):
				cv2.imwrite('data/cv_save.jpg', images)
				print("image saved.")
	finally:
		pipeline.stop()


if __name__ == '__main__':
	matcher = TemplateMatcher(MIN_MATCH_COUNT=15, select_threshold=0.9)
	matcher.set_template_dir(matcher.tpl_path + "template\\")
	test(matcher)
