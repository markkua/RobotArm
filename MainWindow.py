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
        MainWindow.resize(1230, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.exitButton = QtWidgets.QPushButton(self.centralwidget)
        self.exitButton.setGeometry(QtCore.QRect(0, 643, 93, 28))
        self.exitButton.setObjectName("exitButton")
        self.startButton = QtWidgets.QPushButton(self.centralwidget)
        self.startButton.setGeometry(QtCore.QRect(30, 60, 121, 51))
        self.startButton.setObjectName("startButton")
        self.imgLabel_r = QtWidgets.QLabel(self.centralwidget)
        self.imgLabel_r.setGeometry(QtCore.QRect(240, 60, 960, 540))
        self.imgLabel_r.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.imgLabel_r.setObjectName("imgLabel_r")
        self.buttonVoice = QtWidgets.QPushButton(self.centralwidget)
        self.buttonVoice.setGeometry(QtCore.QRect(30, 200, 121, 81))
        self.buttonVoice.setObjectName("buttonVoice")
        self.groupBoxCamera = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxCamera.setGeometry(QtCore.QRect(20, 350, 201, 231))
        self.groupBoxCamera.setObjectName("groupBoxCamera")
        self.thresholdSlider = QtWidgets.QSlider(self.groupBoxCamera)
        self.thresholdSlider.setGeometry(QtCore.QRect(30, 50, 131, 22))
        self.thresholdSlider.setMaximum(255)
        self.thresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholdSlider.setObjectName("thresholdSlider")
        self.slider_lable = QtWidgets.QLabel(self.groupBoxCamera)
        self.slider_lable.setGeometry(QtCore.QRect(50, 70, 111, 31))
        self.slider_lable.setObjectName("slider_lable")
        self.slider_lable_2 = QtWidgets.QLabel(self.groupBoxCamera)
        self.slider_lable_2.setGeometry(QtCore.QRect(10, 20, 111, 31))
        self.slider_lable_2.setObjectName("slider_lable_2")
        self.buttonCalibrate = QtWidgets.QPushButton(self.groupBoxCamera)
        self.buttonCalibrate.setGeometry(QtCore.QRect(80, 180, 121, 41))
        self.buttonCalibrate.setObjectName("buttonCalibrate")
        # MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1230, 26))
        self.menubar.setDefaultUp(True)
        self.menubar.setObjectName("menubar")
        # MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        # MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Monitor"))
        self.exitButton.setText(_translate("MainWindow", "exit"))
        self.startButton.setText(_translate("MainWindow", "Start"))
        self.imgLabel_r.setText(_translate("MainWindow", "ImgLabel"))
        self.buttonVoice.setText(_translate("MainWindow", "Voice Control"))
        self.groupBoxCamera.setTitle(_translate("MainWindow", "Camera"))
        self.slider_lable.setText(_translate("MainWindow", "TextLabel"))
        self.slider_lable_2.setText(_translate("MainWindow", "TextLabel"))
        self.buttonCalibrate.setText(_translate("MainWindow", "Calibrate"))

