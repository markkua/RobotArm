# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from pyzbar.pyzbar import decode
from RealsenseManager import RealsenseManager
import pyzbar.pyzbar as pyzbar


def detect_qrcode(image):
	barcodes = pyzbar.decode(image)
	
	for barcode in barcodes:
		# 提取条形码的边界框的位置
		# 画出图像中条形码的边界框
		(x, y, w, h) = barcode.rect
		
		points = []
		for point in barcode.polygon:
			points.append([point.x, point.y])
		
		points = np.array(points)
		points = points.reshape((1, -1, 2))
		print(points)
		
		cv2.polylines(image, points, True, (100, 100, 200), 1)
		
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
		
		# 条形码数据为字节对象，所以如果我们想在输出图像上
		# 画出来，就需要先将它转换成字符串
		barcodeData = barcode.data.decode("utf-8")
		barcodeType = barcode.type
		
		# 绘出图像上条形码的数据和条形码类型
		text = "{} ({})".format(barcodeData, barcodeType)
		cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
		            .5, (0, 0, 125), 2)
		
		return image


if __name__ == '__main__':
	realsense = RealsenseManager(color_resolution=(640, 480), depth_resolution=(640, 360))
	# realsense.view_color_image()
	
	flag = True
	while flag:
		
		image = realsense.get_color_image_from_frames(realsense.get_aligned_frames())
		
		image = detect_qrcode(image)
			
		cv2.imshow("QRcode", image)
		key = cv2.waitKey(1)
		if ord('q') == key:
			flag = False
