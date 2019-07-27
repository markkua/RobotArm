# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import time
import os
from typing import List
import pyrealsense2 as rs


class TemplateMatcher:
	
	# 公共变量
	tpl_path = ""  # template_path
	tpl_files = []  # template_files
	
	def __init__(self, ):
		# init data path
		cwd = os.getcwd()
		self.set_template_dir(cwd + '\\positioning_data\\template\\')
		
	def set_template_dir(self, data_dir: str):
		self.tpl_path = data_dir
		self.tpl_files.clear()
		for file in os.listdir(self.tpl_path):
			if file.lower().endswith('.jpg'):
				self.tpl_files.append(file)
		if 0 == len(self.tpl_files):
			print("No template image! Please check the path!")
	
	def match_templates(self, color_image, draw_polygon=True, MIN_MATCH_COUNT=15, select_threshold=0.9):
		start_time = time.time()
		match_result = []  # ['point_name', [x, y], 置信指数]
	
		result_image = color_image
		try:
			# Initiate SIFT detector创建sift检测器
			sift = cv2.xfeatures2d.SIFT_create()
			
			# find the keypoints and descriptors with SIFT
			key_point_image, des_image = sift.detectAndCompute(color_image, None)
			# print("Find %d key points in image" % key_point_image.__len__())
			
			# 逐个模板处理
			for img_file in self.tpl_files:
				# Read image
				filename = self.tpl_path + img_file
				template_img = cv2.imread(filename, cv2.IMREAD_COLOR)
				
				# SIFT
				key_point, des = sift.detectAndCompute(template_img, None)
				
				# 创建设置FLANN匹配
				FLANN_INDEX_KDTREE = 0
				
				# 匹配算法
				index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
				# 搜索次数
				search_params = dict(checks=30)  # TODO 调参
				flann = cv2.FlannBasedMatcher(index_params, search_params)
				# knnMatch
				matches = flann.knnMatch(des_image, des, k=2)
				# print("Find %d FLANN matches" % matches.__len__())
				
				# store all the good matches as per Lowe's ratio test.
				good_match = []
				# 舍弃大于threshold的匹配, threshold越小越严格
				for m, n in matches:
					if m.distance < select_threshold * n.distance:
						good_match.append(m)
				
				if len(good_match) > MIN_MATCH_COUNT:
					# 认为找到了目标
					print('\033[32mMatch %s with %d good match points！\033[0m' % (img_file, good_match.__len__()))
					# 获取关键点的坐标
					src_pts = np.float32([key_point_image[m.queryIdx].pt for m in good_match]).reshape(-1, 1, 2)
					dst_pts = np.float32([key_point[m.trainIdx].pt for m in good_match]).reshape(-1, 1, 2)
					# 计算变换矩阵和MASK
					M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
					# matchesMask = mask.ravel().tolist()
					h, w = template_img.shape[:2]
					# 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
					corner_pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
					corner_dst = cv2.perspectiveTransform(corner_pts, M)
					# 画出四角
					if draw_polygon:
						result_image = cv2.polylines(result_image, [np.int32(corner_dst)], True, [0, 0, 255], 2, cv2.LINE_AA)
						print("polygon drawn")
					match_result.append([img_file, corner_dst, len(good_match)])
					
					# TODO 根据mask筛选关键点dst，画出关键点   这关键点是错的啊，

					# TODO 计算置信指数
					
					# 返回结果
					
				else:
					print("Not enough matches are found - %d/%d" % (len(good_match), MIN_MATCH_COUNT))
					# matchesMask = None
		
		# except IOError as e:
		# 	print("IOERROR", e.__str__())
		except cv2.error as e:
			print('\033[31mCV2ERROR！\033[0m', e.__str__())
		finally:
			end_time = time.time()
			print("Matching time: ", end_time - start_time)
		
		return result_image, match_result  # TODO 返回坐标


def remove_background(clipping_distance_in_meters, depth_scale: float,
                      depth_image, color_image):
	clipping_distance = clipping_distance_in_meters / depth_scale
	
	grey_color = 153
	depth_image_3d = np.dstack(
		(depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
	bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
	return bg_removed


def test():
	matcher = TemplateMatcher()
	
	# Create a pipeline
	pipeline = rs.pipeline()
	
	# Create a config and configure the pipeline to stream
	#  different resolutions of color and depth streams
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
	
	# Start streaming
	profile = pipeline.start(config)
	
	# Getting the depth sensor's depth scale (see rs-align example for explanation)
	depth_sensor = profile.get_device().first_depth_sensor()
	# depth_sensor.set_option(rs.RS2_OPTION_VISUAL_PRESET, rs.RS2_RS400_VISUAL_PRESET_HIGH_ACCURACY)
	
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
	while True:
		try:
			# Get frameset of color and depth
			frames = pipeline.wait_for_frames()
			# frames.get_depth_frame() is a 640x360 depth image
			
			# Align the depth frame to color frame
			aligned_frames = align.process(frames)
			
			depth_frame = aligned_frames.get_depth_frame()
			color_frame = aligned_frames.get_color_frame()
			
			# Validate that frames are valid
			if not color_frame:
				return
			
			depth_image = np.asanyarray(depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())
			
			# 清除背景
			# if depth_scale:
			# 	color_image = remove_background(1, depth_scale, depth_image, color_image)
			
			""""""
			images, match_result = matcher.match_templates(color_image, select_threshold=0.7, MIN_MATCH_COUNT=20)

			cv2.namedWindow('Match', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Match', images)
			
			if 0 != match_result.__len__():
				cv2.waitKey(2000)
			
			key = cv2.waitKey(1)
			# Press esc or 'q' to close the image window
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				print('\033[31mQuit.\033[0m')
				break
			elif key == ord('s'):
				cv2.imwrite('data/cv_save.jpg', images)
				print("image saved.")
			
		except cv2.error as e:
			print('\033[31mCV2ERROR: %s \033[0m' % (e.__str__()))
			break
		finally:
			pass
	
	pipeline.stop()
	cv2.destroyAllWindows()


def test_2():
	matcher = TemplateMatcher()
	cap = cv2.VideoCapture(0)
	while True:
		ret, image = cap.read()
		final_image, match_result = matcher.match_templates(image, MIN_MATCH_COUNT=20, select_threshold=0.8)
		print("match result =", match_result)
		cv2.imshow("Result", final_image)
		key = cv2.waitKey(1)
		if ord('q') == key:
			break
	output = 'data/cv_result.jpg'
	cv2.imwrite(output, final_image)
	cv2.destroyAllWindows()


if __name__ == '__main__':
	test()
	# test_2()