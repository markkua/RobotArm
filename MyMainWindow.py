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
	def __init__(self):
		# 初始化窗口
		super().__init__()
		self.setupUi(self)
		
		# 定义线程
		self.mainThread = MainThread()
		self.monitor = MonitorThread(self.mainThread)
		
		# 绑定信号
		self.monitor.img_signal.connect(self._on_update_image_display)
		self.monitor.voice_ready_signal.connect(self._on_voice_ready)
		
		self.startButton.clicked.connect(self._on_start_button_clicked)
		self.buttonVoice.clicked.connect(self._on_voice_button_clicked)
		self.buttonCalibrate.clicked.connect(self._on_calibrate_button_clicked)
		self.exitButton.clicked.connect(self._on_exit_button_clicked)

		# 启动monitor
		self.monitor.start()
		
		self.buttonVoice.setEnabled(False)

		# self._on_start_button_clicked()
		self.show()
		Printer.print("Window ready", Printer.green)
		
	def _on_start_button_clicked(self):
		print('start button clicked')
		self.startButton.setEnabled(False)
		
		self.mainThread.start()
		Printer.print('Main thread started.', Printer.green)
		
		self.startButton.setText('Started')
		
		self.buttonVoice.setEnabled(True)
		QApplication.processEvents()
		
		
	def _on_voice_button_clicked(self):
		print('voice button clicked')
		# TODO set flag（先判断）
		if self.mainThread.isVoiceGet == 0:
			self.mainThread.audioT()
			self.buttonVoice.setEnabled(False)
		else:
			pass
	
	def _on_calibrate_button_clicked(self):
		print('calibrate button clicked')
		# TODO set flag
		self.mainThread.needCali = 1
	
	def _on_exit_button_clicked(self):
		self.mainThread.quit()  # 退出线程  TODO 改成MainThread的
		self.monitor.quit()
		self.close()  # 关闭窗口
		Printer.print("exit", Printer.green)
	
	def _on_voice_ready(self):
		self.buttonVoice.setEnabled(True)
		
	def _on_update_image_display(self, image):
		# print("update image")
		try:
			if isinstance(image, int):
				return
			w = self.imgLabel_r.width()
			h = self.imgLabel_r.height()
			print('w=%d, h=%h' % (w, h))
			cv2.resize(image, (w, h))
			bytesPerLine = 3 * w
			self.qImg = QImage(image.data, w, h, bytesPerLine,
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
			
			QApplication.processEvents()
			time.sleep(0.1)

#
# class qMainThread(QThread, MainThread):
#
# 	def run(self):
# 		print('preparing main thread...')
# 		self.init()
# 		Printer.print('Main thread ready.', Printer.green)
# 		self.startT()
#

if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MyMainWindow()
	# window.show()
	# window.detect_test()
	sys.exit(app.exec_())

