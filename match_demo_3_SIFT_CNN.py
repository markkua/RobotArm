# !/usr/bin/python3

import cv2


def sift_alignment(image_1: str, image_2: str):
	"""
		Aligns two images by using the SIFT features.

		Step 1. The function first detects the SIFT features in I1 and I2.
		Step 2. Then it uses match(I1,I2) function to find the matched pairs between
		the two images.
		Step 3. The matched pairs returned by Step 2 are potential matches based
		on similarity of local appearance, many of which may be incorrect.
		Therefore, we do a ratio test to find the good matches.

		Reference: https://docs.opencv.org/3.4.3/dc/dc3/tutorial_py_matcher.html

		Parameters:
			image_1, image_2: filename as string
		Returns:
			(matched pairs number, good matched pairs number, match_image)
	"""
	im1 = cv2.imread(image_1, cv2.IMREAD_COLOR)
	im2 = cv2.imread(image_2, cv2.IMREAD_COLOR)
	
	sift = cv2.xfeatures2d.SIFT_create()
	key_points_1, descriptors_1 = sift.detectAndCompute(im1, None)
	key_points_2, descriptors_2 = sift.detectAndCompute(im2, None)
	
	bf_matcher = cv2.BFMatcher()  # brute force matcher
	# matches = bf_matcher.match(descriptors_1, descriptors_2)  # result is not good
	matches = bf_matcher.knnMatch(descriptors_1, descriptors_2, k=2)
	
	# Apply ratio test
	good_matches = []
	for m, n in matches:
		if m.distance < 0.6 * n.distance:  # this parameter affects the result filtering
			good_matches.append([m])
	
	match_img = cv2.drawMatchesKnn(im1, key_points_1, im2, key_points_2,
	                               good_matches, None, flags=2)
	return len(matches), len(good_matches), match_img


if __name__ == '__main__':
	matches, good_matches, match_img = sift_alignment('data/img-2-2.jpg', 'data/template-2.jpg')
	cv2.imwrite('match.png', match_img)