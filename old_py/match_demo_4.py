# -*- encoding: utf-8 -*-

# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
import time


def template_match(template_img, target_img, output_imgname=None, threshold=0.7):
	print("start matching:")
	MIN_MATCH_COUNT = 20  # 设置最低特征点匹配数量为10
	
	# 缩小尺寸
	# template_img = resize_img(template_img, 0.5)
	# target_img = resize_img(target_img, 0.5)
	
	start_time = time.time()
	
	# TODO try,expect cv2.error
	
	# Initiate SIFT detector创建sift检测器
	sift = cv2.xfeatures2d.SIFT_create()
	
	# find the keypoints and descriptors with SIFT
	key_point1, des1 = sift.detectAndCompute(template_img, None)
	key_point2, des2 = sift.detectAndCompute(target_img, None)
	print("Find %d key points in template" % key_point1.__len__())
	# TODO
	print("Find %d key points in target" % key_point2.__len__())
	
	# 创建设置FLANN匹配
	FLANN_INDEX_KDTREE = 0
	# 匹配算法
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	# 搜索次数
	search_params = dict(checks=100)  # TODO checks=10
	flann = cv2.FlannBasedMatcher(index_params, search_params)
	
	matches = flann.knnMatch(des1, des2, k=2)
	print("Find %d FLANN matches" % matches.__len__())
	
	# store all the good matches as per Lowe's ratio test.
	good_match = []
	# 舍弃大于threshold的匹配, threshold越小越严格
	for m, n in matches:
		if m.distance < threshold * n.distance:
			good_match.append(m)
	print("Find %d good match points with threshold=%.3f" % (good_match.__len__(), threshold))
	
	if len(good_match) > MIN_MATCH_COUNT:
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
		cv2.polylines(target_img, [np.int32(dst)], True, [0, 0, 255], 2, cv2.LINE_AA)
		
		result_image = cv2.polylines(target_img, [np.int32(dst)], True, [0, 0, 255], 2, cv2.LINE_AA)
		
		# match_pts_dst = []
		# ave_x, ave_y = 0, 0
		# for i in range(len(dst_pts)):
		# 	if 1 == mask[i]:
		# 		match_pts_dst.append(dst_pts[i][0])
		# 		ave_x += dst_pts[i][0][0]
		# 		ave_y += dst_pts[i][0][1]
		# 		# TODO 画出关键点
		# 		cv2.circle(target_img, (dst_pts[i][0][0], dst_pts[i][0][0]), 3, (0, 0, 255))
		# center_point = np.asarray([ave_x, ave_y])

		draw_params = dict(matchColor=(255, 0, 0),
						   # singlePointColor=(0, 0, 255),
						   matchesMask=matchesMask,
						   flags=2)
		# result_image = cv2.drawMatches(template_img, key_point1, target_img, key_point2, good_match, None,
		#                                **draw_params)
	
	else:
		print("Not enough matches are found - %d/%d" % (len(good_match), MIN_MATCH_COUNT))
		result_image = target_img
	
	end_time = time.time()
	print("Matching time: ", end_time - start_time)
	
	return result_image


# return target


def resize_img(img, percentage):
	h, w = img.shape[:2]
	result = cv2.resize(img, (int(w * percentage), int(h * percentage)))
	return result


if __name__ == '__main__':
	template_image = cv2.imread("positioning_data/template/red_star.jpg", cv2.IMREAD_COLOR)
	template_image = resize_img(template_image, 0.5)
	
	import pyrealsense2 as rs
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
	
			# Stack both images horizontally
			# images = np.hstack((color_image, depth_colormap))
			
			images = template_match(template_image, color_image, threshold=0.75)
	
			# Show images
			cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('RealSense', images)
			cv2.waitKey(1)
	
	finally:
		
		# Stop streaming
		pipeline.stop()
	
	#
	# cap = cv2.VideoCapture(0)
	#
	# # 为保存视频做准备
	# while True:
	# 	ret, image = cap.read()
	#
	# 	image = template_match(template_image, image, threshold=0.75)
	#
	# 	cv2.imshow("Result", image)
	# 	key = cv2.waitKey(1)
	# 	if ord('q') == key:
	# 		break
	# 	elif ord('s') == key:
	# 		cv2.imwrite("positioning_data/result.jpg", image)
	# # 释放资源
	# cap.release()
	# cv2.destroyAllWindows()

