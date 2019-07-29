# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import sys
import time

from ImgProcessor import ImgProcessor
from ConsolePrinter import Printer
from RealsenseManager import RealsenseManager

from MainWindow import Ui_MainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget


class MainThread(QThread):
	img_signal = pyqtSignal(object)  # 输出图像的信号
	
	realsense = RealsenseManager()
	
	# frame_start_time = 0
	
	"""控制的主线程"""
	def __init__(self):
		super().__init__()
		
	def run(self):
		"""控制主循环"""
		while True:
			# self.frame_start_time = time.time()
			frames = self.realsense.get_aligned_frames()
			color_image = self.realsense.get_color_image_from_frames(frames)
			
			
			# 识别目标
			
			# 识别控制点
			
			# 解相机位置
			
			# 坐标转换
			
			# 控制抓取
			
			# 控制移动
			
			# 控制释放
			
			result_image = color_image
			self.img_signal.emit(result_image)
			# print("frame rate=", 1 / (time.time() - self.frame_start_time))
	
class ImgShowThread(QThread):
	"""	用于测试窗体显示的线程 """
	
	def __init__(self):
		super().__init__()
	
	def run(self):
		self._run_from_webcam()
	
	def _run_from_webcam(self):
		cap = cv2.VideoCapture(0)
		while True:
			ret, image = cap.read()
			self.img_signal.emit(image)


class PickColorThread(QThread):
	"""
	单色
	"""
	img_signal = pyqtSignal(object)  # 输出图像的信号
	
	value = 0
	
	def __init__(self):
		super().__init__()
		print("thread ready")
	
	def __del__(self):
		print("PickColorThread quit")
	
	def run(self):
		self._run_from_webcam()
	
	def _run_from_webcam(self):
		cap = cv2.VideoCapture(0)
		while True:
			ret, image = cap.read()
			image = ImgProcessor.pick_color(image, self.value, 10)
			self.img_signal.emit(image)

