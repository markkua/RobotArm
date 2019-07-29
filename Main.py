# -*- encoding: utf-8 -*-

import numpy as np
import cv2
import sys

from ImgProcessor import ImgProcessor
from ConsolePrinter import Printer
from Threads import *

import time

from MainWindow import Ui_MainWindow
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget


class MyMainWindow(QWidget, Ui_MainWindow):
	
	def __init__(self):
		# 初始化窗口
		super().__init__()
		self.setupUi(self)

		# 定义线程
		self.main_thread = MainThread()  # 主线程
		
		# 绑定信号
		# self.running_img_thread.img_signal.connect(self._update_image)
		self.main_thread.img_signal.connect(self._update_image_display)
		# self.startButton.clicked.connect(self.running_img_thread.start)
		self.exitButton.clicked.connect(self._on_exit_button_clicked)
		self.thresholdSlider.valueChanged.connect(self._on_slider_change_value)
		
		self.thresholdSlider.setValue(int(self.thresholdSlider.maximum() / 2))
		
		Printer.print("MainWindow ready", Printer.green)
		
		self.main_thread.start()
		Printer.print("Main thread start", Printer.green)
	
	def _on_exit_button_clicked(self):
		self.main_thread.quit()  # 退出线程
		self.close()  # 关闭窗口
		Printer.print("exit", Printer.green)
		
	def _update_image_display(self, image):
		# print("update image")
		height, width, channel = image.shape
		bytesPerLine = 3 * width
		self.qImg = QImage(image.data, width, height, bytesPerLine,
		                   QImage.Format_RGB888).rgbSwapped()
		# 将Qimage显示出来
		self.imgLabel_r.setPixmap(QPixmap.fromImage(self.qImg))
		
	def _on_slider_change_value(self, value):
		self.slider_lable.setText(value.__str__())


if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MyMainWindow()
	window.show()
	sys.exit(app.exec_())

