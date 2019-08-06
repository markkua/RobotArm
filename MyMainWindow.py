# -*- encoding: utf-8 -*-

from MainWindow import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from ConsolePrinter import Printer
from mainThread import MainThread
import sys
import time
import cv2


class MyMainWindow(QWidget, Ui_MainWindow):
	window_ready_signal = pyqtSignal()
	
	def __init__(self):
		# 初始化窗口
		super().__init__()
		self.setupUi(self)
		
		# 推迟主线程初始化到窗体显示后
		self.startButton.setEnabled(False)
		self.startButton.setText('MainThread Preparing...')
		
		# 定义线程
		self.mainThread = MainThread()  # TODO 运行前注释掉, 启用下一行
		# self.mainThread = QThread()  # 后面初始化
		self.monitor = MonitorThread(self.mainThread)
		
		# 绑定信号
		self.monitor.img_signal.connect(self._on_update_image_display)
		self.monitor.voice_ready_signal.connect(self._on_voice_ready)
		self.window_ready_signal.connect(self._init_main_thread)
		
		self.startButton.clicked.connect(self._on_start_button_clicked)
		self.buttonVoice.clicked.connect(self._on_voice_button_clicked)
		self.buttonCalibrate.clicked.connect(self._on_calibrate_button_clicked)
		self.exitButton.clicked.connect(self._on_exit_button_clicked)

		Printer.print("Window ready", Printer.green)
		self.window_ready_signal.emit()
		
	def _init_main_thread(self):
		# self.mainThread = MainThread()
		self.startButton.setText('Start')
		Printer.print('MainThread Ready', Printer.green)
		self.startButton.setEnabled(True)
		# 启动monitor
		self.monitor.start()
		
	def _on_start_button_clicked(self):
		print('start button clicked')
		self.mainThread.start()
		Printer.print('Main thread started.', Printer.green)
		self.startButton.setText('Started')
		self.startButton.setEnabled(False)
		
	def _on_voice_button_clicked(self):
		print('voice button clicked')
		# TODO set flag（先判断）
		if self.mainThread.isVoiceGet == 0:
			
			self.buttonVoice.setEnabled(False)
			return
		else:
			return
	
	def _on_calibrate_button_clicked(self):
		print('calibrate button clicked')
		# TODO set flag
		self.mainThread.needCali = 1
	
	def _on_exit_button_clicked(self):
		# self.mainThread.quit()  # 退出线程  TODO 改成MainThread的
		self.monitor.quit()
		self.close()  # 关闭窗口
		Printer.print("exit", Printer.green)
	
	def _on_voice_ready(self):
		self.buttonVoice.setEnabled(True)
		
	def _on_update_image_display(self, image):
		# print("update image")
		try:
			cv2.resize(image, (960, 540))
			height, width, channel = image.shape
			bytesPerLine = 3 * width
			self.qImg = QImage(image.data, width, height, bytesPerLine,
			                   QImage.Format_RGB888).rgbSwapped()
			# 将Qimage显示出来
			self.imgLabel_r.setPixmap(QPixmap.fromImage(self.qImg))
		except Exception as e:
			Printer.print('update image error: ' + e.__str__(), Printer.red)
		
	def _on_slider_change_value(self, value):
		self.slider_lable.setText('%f' % (self.thresholdSlider.value() / 255))
		self.main_thread.value = self.thresholdSlider.value() / 255
	

class MonitorThread(QThread):

	img_signal = pyqtSignal(object)  # 输出图像的信号
	
	voice_ready_signal = pyqtSignal()  # Voice Ready的信号
	
	def __init__(self, mainThread):
		super().__init__()
		self.mainThread = mainThread
		
	def run(self) -> None:
		while True:
			# 获取图片
			img = self.mainThread.marked_img
			self.img_signal.emit(img)
			
			# 判断voice ready
			if self.mainThread.isVoiceGet == 0:
				self.voice_ready_signal.emit()
			
			
			time.sleep(0.2)
		

if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MyMainWindow()
	window.show()
	# window.detect_test()
	sys.exit(app.exec_())

