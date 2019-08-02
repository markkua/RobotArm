# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow._resize(1080, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.thresholdSlider = QtWidgets.QSlider(self.centralwidget)
        self.thresholdSlider.setGeometry(QtCore.QRect(720, 613, 160, 22))
        self.thresholdSlider.setMaximum(255)
        self.thresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholdSlider.setObjectName("thresholdSlider")
        self.slider_lable = QtWidgets.QLabel(self.centralwidget)
        self.slider_lable.setGeometry(QtCore.QRect(760, 633, 111, 31))
        self.slider_lable.setObjectName("slider_lable")
        self.exitButton = QtWidgets.QPushButton(self.centralwidget)
        self.exitButton.setGeometry(QtCore.QRect(0, 643, 93, 28))
        self.exitButton.setObjectName("exitButton")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(960, 590, 93, 51))
        self.startButton.setObjectName("startButton")
        self.imgLabel_r = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel_r.setGeometry(QtCore.QRect(10, 30, 1039, 489))
        self.imgLabel_r.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.imgLabel_r.setObjectName("imgLabel_r")
        # MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 26))
        self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ArmMonitor"))
        self.slider_lable.setText(_translate("MainWindow", "TextLabel"))
        self.exitButton.setText(_translate("MainWindow", "exit"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.imgLabel_r.setText(_translate("MainWindow", "ImgLabel"))

