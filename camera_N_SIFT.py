# -*- encoding: utf-8 -*-

# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
import time

import pyrealsense2 as rs


def resize_img(img, percentage):
	if percentage == 1:
		return img
	h, w = img.shape[:2]
	result = cv2.resize(img, (int(w * percentage), int(h * percentage)))
	return result


def template_match(template, target, output_imgname=None, threshold=0.7):
	print("start matching:")
	MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10
	
	# 缩小尺寸
	template = resize_img(template, 0.5)
	target = resize_img(target, 1)
	
	start_time = time.time()
	
	# Initiate SIFT detector创建sift检测器
	sift = cv2.xfeatures2d.SIFT_create()
	
	# find the keypoints and descriptors with SIFT
	key_point1, des1 = sift.detectAndCompute(template, None)
	key_point2, des2 = sift.detectAndCompute(target, None)
	print("Find %d key points in template" % key_point1.__len__())
	print("Find %d key points in target" % key_point2.__len__())
	
	MIN_KEY_POINT_NUM = 10
	if key_point1.__len__() < MIN_KEY_POINT_NUM or key_point2.__len__() < MIN_KEY_POINT_NUM:
		return target
	
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
	good = []
	
	# 舍弃大于threshold的匹配, threshold越小越严格
	for m, n in matches:
		if m.distance < threshold * n.distance:
			good.append(m)
	print("Find %d good match points with threshold=%.3f" % (good.__len__(), threshold))
	
	if len(good) > MIN_MATCH_COUNT:
		# 获取关键点的坐标
		src_pts = np.float32([key_point1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
		dst_pts = np.float32([key_point2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
		# 计算变换矩阵和MASK
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()
		h, w = template.shape[:2]
		# 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

		dst = cv2.perspectiveTransform(pts, M)
		
		cv2.polylines(target, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
	else:
		print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
		matchesMask = None
	
	draw_params = dict(matchColor=(255, 0, 0),  # BGR ??
	                   singlePointColor=None,
	                   matchesMask=matchesMask,
	                   flags=2)
	
	end_time = time.time()
	print("Matching time: ", end_time - start_time)
	
	result = cv2.drawMatches(template, key_point1, target, key_point2, good, None, **draw_params)
	
	print("Done.")
	
	return result


def camera_loop(template, threshold=0.7):
	## License: Apache 2.0. See LICENSE file in root directory.
	## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
	
	###############################################
	##      Open CV and Numpy integration        ##
	###############################################

	
	# Configure depth and color streams
	pipeline = rs.pipeline()
	config = rs.config()
	config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
	config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
	
	# Start streaming
	pipeline.start(config)
	
	try:
		while True:
			# Wait for a coherent pair of frames: depth and color
			frames = pipeline.wait_for_frames()
			depth_frame = frames.get_depth_frame()
			color_frame = frames.get_color_frame()
			if not depth_frame or not color_frame:
				continue
			
			# Convert images to numpy arrays
			depth_image = np.asanyarray(depth_frame.get_data())
			color_image = np.asanyarray(color_frame.get_data())
			
			# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
			depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
			
			# images = template_match(template, color_image, threshold)
			
			# Show images
			cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('RealSense', images)
			cv2.waitKey(1)
	
	finally:
		# Stop streaming
		pipeline.stop()


if __name__ == '__main__':
	template_imgname = 'data/template-2.jpg'
	
	template = cv2.imread(template_imgname, cv2.IMREAD_COLOR)  # queryImage
	
	camera_loop(template, threshold=0.8)
