# -*- encoding: utf-8 -*-

import cv2

from realsenseCamera_old import RealsenseCamera
from ToolDetect import *
from SerialPart import *
from speech.VoiceRecognition import *

from PyQt5.QtCore import QThread, pyqtSignal


class MainThread(QThread):
	img_signal = pyqtSignal(object)  # 输出图像的信号
	
	realsenseCamera = RealsenseCamera()
	
	value = 0  # 用来找阈值
	
	robotArm = RobotArm('COM3')
	
	voiceRecognition = VoiceRecognition()
	
	# frame_start_time = 0
	
	"""控制的主线程"""
	def __init__(self):
		pass
		super().__init__()
		# self.realsenseCamera.load_control_point_file()
		# self.model = create_model('mask_rcnn_tools_0030.h5')

	def detect_tool(self, image):
		r = self.model.detect([image], verbose=1)[0]
		cout, list = calculate(r)
		return cout, list
	
	def voice(self):
		self.voiceRecognition.run()
		
	def run(self):
		"""
		控制主循环
		用同一帧进行处理
		"""
		
		self.robotArm.moveObject([25.0, 10.0, 1.0], [25.0, -10.0, 1.0], 1.0)
		
		while True:
			# self.frame_start_time = time.time()
			
			aligned_frames = self.realsenseCamera.get_aligned_frames()
			color_image = self.realsenseCamera.get_color_image_from_frames(aligned_frames)
			# print("get new frame")
			
			result_image = color_image.copy()
			
			# print(color_image)
			
			# 相机定位，定位成功了再去识别
			result_image = self.realsenseCamera.test(color_image)
			# self.realsenseCamera.get_transform_matrix(aligned_frames)
			#
			# if self.realsenseCamera.if_get_position:
			# 	Printer.print("camera located.", Printer.green)
			# else:
			# 	continue
				
			# 识别目标
			# cout, target_list = self.detect_tool(color_image)
			# print('target_ls:', target_list)
			# for target in target_list:
			# 	center = target['center']
			# 	cv2.circle(result_image, center, 10, (0, 0, 255))
			#
			# self.img_signal.emit(result_image)

			

			# 坐标转换
			
			# 控制抓取
			
			
			# 控制移动
			
			# 控制释放
			
			# print("frame rate=", 1 / (time.time() - self.frame_start_time))
			
		
	def test_detect(self):
		color_image = cv2.imread('imgdata/detect_test.jpg')
		func = 0  # 0 return pic as well
		cout, list = self.detect_tool(color_image)


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

if __name__ == '__main__':
	thread = MainThread()
	thread.test_detect()
