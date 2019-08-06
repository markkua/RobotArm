import cv2
from SerialPart import *
import threading
from RealseCamera_svd import RealsenseCamera
from ToolDetect import *
from VoiceRecognition import *
from PyQt5.QtCore import QThread, pyqtSignal


class MainThread(QThread):
    img_signal = pyqtSignal(object)  # 输出图像的信号
    realsenseCamera = RealsenseCamera()
    aligned_frames = 0      # a bunch of images
    color_image = 0         # colorful images
    calied_img = 0
    marked_img = 0
    value = 0  # 用来找阈值
    isVoiceGet = 0

    #Coordinate Transfer
    needCali = 0

    # Recognize model
    model = 0
    recognizeResult = 0
    recCount = 0
    recList = 0

    # Workspace Area
    pickArea = (11, 10.5, 30, 26)
    toolBox = ({'id':1, 'name':'screwdriver', 'pos':(15, -24), 'wide':0.0},
               {'id':2, 'name':'pincer', 'pos':(25, -11), 'wide':0.5},
               {'id':3, 'name':'alien', 'pos':(25, 0), 'wide':4.0},
               {'id':4, 'name':'pen', 'pos':(25, 0), 'wide':0.0})
    toolInHand = 'BY1'

    # robotArm
    robotArm = RobotArm('COM3')

    #voice recognition
    voiceRecognition = VoiceRecognition()

    exitFlag = 0
    # init: load data
    def init(self):

        self.aligned_frames = self.realsenseCamera.get_aligned_frames()
        self.marked_img = self.calied_img = self.color_image = self.realsenseCamera.get_color_image_from_frames(self.aligned_frames)
        self.model = create_model("mask_rcnn_tools_0030.h5")

        self._camCalibrate()
        # lack: Transform, voicRec
        pass

    # stop: Actually there's no way to stop
    def stop(self):
        self.exitFlag = 1
        pass

    # start: Start the main Thread
    def run(self):
        print('running...')
        self.init()
        print('init done')
        while self.exitFlag != 1:
            # 1. Motion Detection
            last_color_image = self.color_image
            color_image = self.realsenseCamera.get_color_image_from_frames(self.realsenseCamera.get_aligned_frames())
            while (self._ifMoving(last_color_image, color_image, 6, 2800) == False and self.isVoiceGet == 0):   # fine tune the kern and Thres to get best result
                last_color_image = color_image
                color_image = self.realsenseCamera.get_color_image_from_frames(self.realsenseCamera.get_aligned_frames())

            if(self.isVoiceGet == 0):
                while (self._ifMoving(last_color_image, color_image, 6, 800) == True):
                    last_color_image = color_image
                    a_img = self.realsenseCamera.get_aligned_frames()
                    color_image = self.realsenseCamera.get_color_image_from_frames(a_img)
                time.sleep(0.5)
                self.aligned_frames = self.realsenseCamera.get_aligned_frames()
                self.color_image = self.realsenseCamera.get_color_image_from_frames(self.aligned_frames)
                print('Start Detect!')

                # 2. Tools Detect
                rgb_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                start_time = time.time()
                self.recognizeResult, self.marked_img = ob_evaluate(model=self.model, image=rgb_image, sampleImg=self.calied_img)
                self.recCount, self.recList = calculate(self.recognizeResult)
                print("time till pick color=", time.time() - start_time)
                print('Detect Finish!')
                print('Central', self._getCentral())

                # 3. get Coordinate
                if (self.needCali):
                    self._camCalibrate()
                realXYZ, self.toolInHand = self._getCentral()
                if(realXYZ[0] != 0 or realXYZ[1] != 0):
                    for i in self.toolBox:
                        if(i['name'] == self.toolInHand):
                            targetXYZ = i['pos']
                            w = i['wide']
                    targetXYZ = (targetXYZ[0], targetXYZ[1], 0.2)
                    print('Coord Obtained')
                    print('---------Finish----------')
                    print('tool is:' + self.toolInHand)
                    print('Actual Coord:', realXYZ)
                    realXYZ[2] = 0.2

                    # 4. Move Arm
                    self.robotArm.moveObject(fromXYZ=realXYZ, toXYZ=targetXYZ, wide=w)
                    self.robotArm.resetArm()

                    print('exec Finish')
                else:
                    print('No Tools Detected!..')

            #---------------------------------------------------------------------------
            else:
                # get voice Message
                self.aligned_frames = self.realsenseCamera.get_aligned_frames()
                self.color_image  = self.realsenseCamera.get_color_image_from_frames(self.aligned_frames)
                print('Voice detected! Start!')

                rgb_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
                start_time = time.time()
                self.recognizeResult, self.marked_img = ob_evaluate(model=self.model, image=rgb_image,
                                                                    sampleImg=self.calied_img)
                self.recCount, self.recList = calculate(self.recognizeResult)
                print("time till pick color=", time.time() - start_time)
                print('Detect Finish!')
                # print('Central', self._getCentral())

                for i in self.toolBox:
                    if i['id'] == self.isVoiceGet:
                        toolName = i['name']
                xyData, self.toolInHand = self._getCentral(toolName)
                if (xyData[0] != 0 or xyData[1] != 0):
                    if (self.needCali):
                        self._camCalibrate()
                    realXYZ = self.realsenseCamera.coor_trans_pixelxy2worldXYZ(self.aligned_frames, xyData)
                    for i in self.toolBox:
                        if (i['name'] == self.toolInHand):
                            targetXYZ = i['pos']
                            w = i['wide']
                    targetXYZ = (targetXYZ[0], targetXYZ[1], 0.2)
                    print('Coord Obtained')
                    print('---------Finish----------')
                    print('Actual Coord:', realXYZ)
                    realXYZ[2] = 0.2

                    # 4. Move Arm
                    self.robotArm.moveObject(fromXYZ=realXYZ, toXYZ=targetXYZ, wide=w)
                    self.robotArm.resetArm()

                    print('exec Finish')
                else:
                    # self.robotArm.swing()
                    print('No Tools Detected!..')
            pass

    # _audioStart: Start the audio detection thread
    def audioT(self):
        # press the Buttum
        # Run this
        self.isVoiceGet = self.voiceRecognition.run()
        pass

    # camCalibrate: Calibrate the cam
    def _camCalibrate(self):
        self.realsenseCamera.get_transform_matrix(self.aligned_frames)
        while(self.realsenseCamera.if_get_position_flag is not True):
            aligned_frames = self.realsenseCamera.get_aligned_frames()
            tmpimg = self.realsenseCamera.get_transform_matrix(aligned_frames)
            
            self.marked_img = tmpimg
        #     cv2.imshow("Positioning", tmpimg)
        #     cv2.waitKey(1)
        # cv2.destroyWindow("Positioning")
        self.needCali = 0
        print("TF_Matrix Obtained!")
        pass

    def _ifMoving(self, frame1, frame2, kern, thres):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        if kern % 2 == 0:
            kern = kern + 1  # 解决滑动条赋值到高斯滤波器是偶数异常抛出
        gray1 = cv2.GaussianBlur(gray1, (kern, kern), 0)
        gray2 = cv2.GaussianBlur(gray2, (kern, kern), 0)
        frameDelta = cv2.absdiff(gray1, gray2)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        flat = False
        for c in cnts:
            if cv2.contourArea(c) > thres:  # 对于较小矩形区域，选择忽略
                flat = True
        return flat
    def _getCentral(self, name=None):
        cent = (0, 0)
        if name==None:
            for i in self.recList:
                c = i['center']
                c = self.realsenseCamera.coor_trans_pixelxy2worldXYZ(self.aligned_frames, c)
                if c is None:
                    c = (0, 0)
                    continue
                if(c[0] > self.pickArea[0] and c[0] < self.pickArea[2] and c[1] > self.pickArea[1] and c[1] < self.pickArea[3]):
                    cent = c
                    name = i['name']
        else:
            for i in self.recList:
                if(i['name'] == name):
                    cent = self.realsenseCamera.coor_trans_pixelxy2worldXYZ(self.aligned_frames, i['center'])
                if cent is None:
                    cent = (0, 0)

        return cent, name

    def _exec(self):
        pass



if __name__ == '__main__':
    r = MainThread()
    r.init()
    r.run()
    r.t1.join()