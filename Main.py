# -*- encoding: utf-8 -*-

from MainWindow import Ui_MainWindow
# from PyQt5 import
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class MyMainWindow(QWidget, Ui_MainWindow):
	def __init__(self):
		# 初始化窗口
		super().__init__()
		self.setupUi(self)
		


if __name__ == '__main__':
	app = QApplication(sys.argv)
	window = MyMainWindow()
	window.show()
	sys.exit(app.exec_())
	
