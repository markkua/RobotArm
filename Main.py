# -*- encoding: utf-8 -*-

from Threads import *

from MainWindow import Ui_MainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget
from ConsolePrinter import Printer


class MyMainWindow(QWidget, Ui_MainWindow):
	
	def __init__(self):
		# 初始化窗口
		super().__init__()
		self.setupUi(self)

		# 定义线程
		self.main_thread = MainThread()  # 主线程
		
		# 绑定信号
		# self.running_img_thread.img_signal.connect(self._update_image)
		self.main_thread.img_signal.connect(self._on_update_image_display)
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
		
	def _on_update_image_display(self, image):
		# print("update image")
		height, width, channel = image.shape
		bytesPerLine = 3 * width
		self.qImg = QImage(image.data, width, height, bytesPerLine,
		                   QImage.Format_RGB888).rgbSwapped()
		# 将Qimage显示出来
		self.imgLabel_r.setPixmap(QPixmap.fromImage(self.qImg))
		
	def _on_slider_change_value(self, value):
		self.slider_lable.setText('%f' % (self.thresholdSlider.value() / 255))
		self.main_thread.value = self.thresholdSlider.value() / 255
		
	def detect_test(self):
		for i in range(20):
			frames = self.realsense.get_aligned_frames()
			color_image = self.realsense.get_color_image_from_frames(frames)
			
			# 识别目标
			ob_detect = Ob_detect()
			model = ob_detect.create_model()
			done_dir = ob_detect.Infer_model(color_image, model)
			
			print("done_dir=", done_dir)


if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MyMainWindow()
	window.show()
	# window.detect_test()
	sys.exit(app.exec_())

