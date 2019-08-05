import numpy as np
from math import sqrt


# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A: np.array, B: np.array):

	N = A.shape[0]
	
	# centroid_A, B
	centroid_A = np.mean(A, axis=0)
	centroid_B = np.mean(B, axis=0)
	
	centroid_A = centroid_A.reshape((1, N))
	centroid_B = centroid_B.reshape((1, N))

	
	AA = A - np.tile(centroid_A, (N, 1))
	BB = B - np.tile(centroid_B, (N, 1))
	
	H = np.matmul(AA.T, BB)
	
	U, S, V = np.linalg.svd(H)
	print('S', S)
	
	R = np.matmul(V.T, U.T)
	
	if np.linalg.det(R) < 0:
		print('Reflection Detected.')
		V[:, 2] *= -1
		R = np.matmul(V.T, U.T)
	
	t = centroid_B.T - np.matmul(R, centroid_A.T)
	
	return R, t


if __name__ == '__main__':
	
	R = np.array([
		[-0.474089119526847, 0.842437682869845, - 0.256004408608807],
		[- 0.513280773442789, - 0.500674161172318, - 0.697042489342198],
		[- 0.715389652119900, - 0.199058119189552, 0.669771237680812]
	], dtype=np.float64)
	t = np.random.random((3, 1))
	
	A = np.random.random((3, 3))
	
	B = np.matmul(R, A.T) + np.tile(t, (1, 3))
	B = B.T
	
	R2, t2 = rigid_transform_3D(A, B)
	
	errR = R2 - R
	errt = t2 - t
	print('errR=', errR)
	print('errt=', errt)
	
	
	
	
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
	# #
	# # A = np.asarray([
	# # 	[14.235121, 25.112, 13.256123],
	# # 	[25.2326, 19.1552, 36.16623],
	# # 	[13.2151, 25.36552, 36.22669]
	# # ])
	# #
	# # B = np.asarray([
	# # 	[104.227490000000, 376.327710716590, 318.766978183650],
	# # 	[172.041690000000, 597.831942815900, 676.272897862500],
	# # 	[172.626210000000, 446.078270287700, 666.463645021100]
	# # ]
	# # )
	#
	# R, t = rigid_transform_3D(A, B)
	# print('R:', R)
	# print('t:', t)
	#
	# err = B - np.matmul(R, A) - t
	# print('err:', err)
	

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