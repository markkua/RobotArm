# -*- encoding: utf-8 -*-

# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
import time


def template_match(template_imgname: str, target_imgname, output_imgname=None, threshold=0.7):
	print("start matching:")
	MIN_MATCH_COUNT = 10  # 设置最低特征点匹配数量为10

	template = cv2.imread(template_imgname, cv2.IMREAD_COLOR)  # queryImage
	target = cv2.imread(target_imgname, cv2.IMREAD_COLOR)  # trainImage
	
	# 缩小尺寸
	template = resize_img(template, 0.5)
	target = resize_img(target, 0.5)
	
	start_time = time.time()
	
	# Initiate SIFT detector创建sift检测器
	sift = cv2.xfeatures2d.SIFT_create()
	
	# find the keypoints and descriptors with SIFT
	key_point1, des1 = sift.detectAndCompute(template, None)
	key_point2, des2 = sift.detectAndCompute(target, None)
	print("Find %d key points in template" % key_point1.__len__())
	# TODO
	print("Find %d key points in target" % key_point2.__len__())
	
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
		
		cv2.polylines(target, [np.int32(dst)], True, [0,0,255], 2, cv2.LINE_AA)
	else:
		print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
		matchesMask = None
		
	# Test
	print("match mask:", matchesMask)
	
	# draw_params = dict(matchColor=(255, 0, 0),  # BGR ??
	#                    singlePointColor=None,
	#                    matchesMask=matchesMask,
	#                    flags=2)
	
	draw_params = dict(matchColor=(255, 0, 0),
	                   # singlePointColor=(0, 0, 255),
	                   matchesMask=matchesMask,
	                   flags=2)

	end_time = time.time()
	print("Matching time: ", end_time-start_time)
	
	result_image = cv2.drawMatches(template, key_point1, target, key_point2, good, None, **draw_params)
	
	print("Done.")
	
	return result_image
	# return target


def resize_img(img, percentage):
	h, w = img.shape[:2]
	result = cv2.resize(img, (int(w * percentage), int(h * percentage)))
	return result


if __name__ == '__main__':
	template = 'data/template-2.jpg'
	target = 'data/img-2-2.jpg'
	output = 'data/result-2-3.jpg'
	final_image = template_match(template, target, output, threshold=0.8)
	cv2.imwrite(output, final_image)
