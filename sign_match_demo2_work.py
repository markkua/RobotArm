# -*- encoding: utf-8 -*-

# 基于FLANN的匹配器(FLANN based Matcher)定位图片
import numpy as np
import cv2
import time
from RealseCamera_before_svd import RealsenseCamera


def match_template(template_img, draw_img, target_kp, target_descrip, threshold, min_match_count=25):
	result_img = draw_img.copy()
	
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
	
	matches = flann.knnMatch(template_des, target_descrip, k=2)
	
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
		matchesMask = mask.ravel().tolist()
		h, w = template_img.shape[:2]
		# 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
		pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
		dst = cv2.perspectiveTransform(pts, M)
		center = np.asarray([np.mean(dst[:, 0, 0]), np.mean(dst[:, 0, 1])])
		print('dst:', dst)
		print('center:', center)
		result_img = cv2.polylines(result_img, [np.int32(dst)], True, [0, 0, 255], 2, cv2.LINE_AA)
		result_img = cv2.circle(result_img, (center[0], center[1]), 3, (0, 255, 0), 3)
	else:
		matchesMask = None
	
	return result_img


def test_match_thread(target_img, temp_img_ls):
	start_time = time.time()
	
	# Initiate SIFT detector创建sift检测器
	sift = cv2.xfeatures2d.SIFT_create()
	
	# compute the descriptors with ORB
	target_kp, target_des = sift.detectAndCompute(target_img, None)
	
	for template in temp_img_ls:
		target_img = match_template(template, target_img, target_kp, target_des, 0.8, min_match_count=15)
	
	print('full_time', time.time() - start_time)
	return target_img


if __name__ == '__main__':
	realsense = RealsenseCamera()
	
	temp_img_ls = [
		cv2.imread('imgdata/matchdata/template1.jpg'),
		# cv2.imread('imgdata/matchdata/template2.jpg'),
		cv2.imread('imgdata/matchdata/template4.jpg'),
		cv2.imread('imgdata/matchdata/template5.jpg')
	]
	
	while True:
		image = realsense.get_color_image_from_frames(realsense.get_aligned_frames())
		
		result_img = test_match_thread(image, temp_img_ls)
		
		cv2.imshow('result', result_img)
		
		key = cv2.waitKey(1)
		if ord('q') == key or 27 == key:
			cv2.destroyAllWindows()
			break
