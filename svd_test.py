import numpy as np
from math import sqrt


# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
	# # centroid_A, B
	# centroid_A = np.mean(A, axis=0)
	# centroid_B = np.mean(B, axis=0)
	#
	# # N = size(A, 1); 点数
	# N = A.shape[0]
	#
	# H = (A - repmat(centroid_A, N, 1))
	# ' * (B - repmat(centroid_B, N, 1));
	#
	# [U, S, V] = svd(H);
	#
	# R = V * U';
	# if det(R) < 0
	# 	printf('Reflection detected\n');
	# V(:, 3) = -1 * V(:, 3);
	# R = V * U';
	# end
	#
	# t = -R * centroid_A
	# ' + centroid_B';
	# detr = det(R)
	assert len(A) == len(B)

	N = A.shape[0]  # total points

	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)

	# centre the points
	AA = A - np.tile(centroid_A, (N, 1))
	BB = B - np.tile(centroid_B, (N, 1))

	# dot is matrix multiplication for array
	H = np.matmul(np.transpose(AA), BB)

	U, S, Vt = np.linalg.svd(H)

	# R = Vt.T * U.T
	R = np.matmul(Vt.T, U.T)
	
	# special reflection case
	if np.linalg.det(R) < 0:
		print("Reflection detected")
		Vt[2, :] *= -1
		R = np.matmul(Vt.T, U.T)

	# t = -R * centroid_A.T + centroid_B.T
	t = np.matmul(-1*R, centroid_B.T)

	return R, t


if __name__ == '__main__':
	# B = np.asarray([
	# 	[-11.071537017822266, 11.890090942382812, -47.500003814697266],
	# 	[0.10266775637865067, 18.987825393676758, -47.500003814697266],
	# 	[0.10266775637865067, 18.987825393676758, -47.500003814697266]
	# ])
	# A = np.asarray([
	# 	[16.2, 16.5, 0.],
	# 	[27.2, 23.5, 0.],
	# 	[33.2, 0.5, 0.]
	# ])
	#
	A = np.asarray([
		[14.235121, 25.112, 13.256123],
		[25.2326, 19.1552, 36.16623],
		[13.2151, 25.36552, 36.22669]
	])
	
	B = np.asarray([
		[104.227490000000, 376.327710716590, 318.766978183650],
		[172.041690000000, 597.831942815900, 676.272897862500],
		[172.626210000000, 446.078270287700, 666.463645021100]
	]
	)
	
	R, t = rigid_transform_3D(A, B)
	print('R:', R)
	print('t:', t)
	
	err = B - np.matmul(R, A) - t
	print('err:', err)
	

"""
found point: {'sign3': [-11.071537017822266, 11.890090942382812, -47.500003814697266], 'sign4': [0.10266775637865067, 18.987825393676758, -47.500003814697266], 'sign5': [6.126349449157715, -4.010044574737549, -48.400001525878906]}
xy3= [1307  208]
xyz3= [-11.071537017822266, 11.890090942382812, -47.500003814697266]
XYZ3= [1.60817362e+01 1.64831940e+01 1.95654209e-03]
xy4= [1516  537]
xyz4= [0.10266775637865067, 18.987825393676758, -47.500003814697266]
XYZ4= [ 2.72852003e+01  2.35348645e+01 -6.46000646e-04]
xy5= [841 711]
xyz5= [-6.126349449157715, -4.010044574737549, -48.400001525878906]
XYZ5= [ 3.32330635e+01  4.81941459e-01 -1.31054144e-03]
"""