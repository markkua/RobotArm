# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import sys
import time

from ImgProcessor import ImgProcessor
from ConsolePrinter import Printer
from RealsenseManager import RealsenseManager
from positioning import Positioning
from Ob_detect import Ob_detect
from RealsenseCamera import RealsenseCamera

from MainWindow import Ui_MainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget


class MainThread(QThread):
	img_signal = pyqtSignal(object)  # 输出图像的信号
	
	realsenseCamera = RealsenseCamera()
	
	# realsense = RealsenseManager()  # 相机控制器
	#
	# positioner = Positioning(realsense)  # 相机定位器
	
	value = 0  # 用来找阈值
	
	# frame_start_time = 0
	
	"""控制的主线程"""
	def __init__(self):
		super().__init__()
		
	def run(self):
		"""
		控制主循环
		用同一帧进行处理
		"""
		while True:
			# self.frame_start_time = time.time()
			
			aligned_frames = self.realsenseCamera.get_aligned_frames()
			color_image = self.realsenseCamera.get_color_image_from_frames(aligned_frames)

			# 相机定位，定位成功了再去识别
			self.realsenseCamera.get_transform_matrix(aligned_frames)
			if not self.realsenseCamera.if_get_position:
				continue
				
			
			# 识别目标
			
			
			# ob_detect = Ob_detect()
			# model = ob_detect.create_model()
			# done_dir = ob_detect.Infer_model(color_image, model)
			#
			# print("done_dir=", done_dir)
			#done_dir包含处理好的图像信息

			# 坐标转换
			
			# 控制抓取
			
			# 控制移动
			
			# 控制释放
			
			# print("frame rate=", 1 / (time.time() - self.frame_start_time))
		

# class ImgShowThread(QThread):
# 	"""	用于测试窗体显示的线程 """
#
# 	def __init__(self):
# 		super().__init__()
#
# 	def run(self):
# 		self._run_from_webcam()
#
# 	def _run_from_webcam(self):
# 		cap = cv2.VideoCapture(0)
# 		while True:
# 			ret, image = cap.read()
# 			self.img_signal.emit(image)
#
#
# class PickColorThread(QThread):
# 	"""
# 	单色
# 	"""
# 	img_signal = pyqtSignal(object)  # 输出图像的信号
#
# 	value = 0
#
# 	def __init__(self):
# 		super().__init__()
# 		print("thread ready")
#
# 	def __del__(self):
# 		print("PickColorThread quit")
#
# 	def run(self):
# 		self._run_from_webcam()
#
# 	def _run_from_webcam(self):
# 		cap = cv2.VideoCapture(0)
# 		while True:
# 			ret, image = cap.read()
# 			image = ImgProcessor.pick_color(image, self.value, 10)
# 			self.img_signal.emit(image)

