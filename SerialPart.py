import serial
from ctypes import *
import time


class Command(Union):
    _fields_ = [("raw", c_byte * 24), ("idat", c_int32 * 6), ("fdat", c_float * 6)]


class RobotArm:
    def __init__(self, portx):
        try:
            self.ser = serial.Serial(port=portx, baudrate=115200, timeout=0.01)
            if (self.ser.is_open):
                pass
        except serial.SerialException as e:
            print("---异常---：", e)

    def __del__(self):
        ret = False
        try:
            self.ser.close()
        except serial.SerialException as e:
            print("---异常---：", e)

    def moveObject(self, fromXYZ, toXYZ, wide, fromvector=[], tovector=[]):
        
        height = 15.0
        
        coord = [fromXYZ[0], fromXYZ[1], height]
        self.moveto(coord, 0.0, 1.0)
        time.sleep(1)
        
        self.release()
        coord = [fromXYZ[0], fromXYZ[1], fromXYZ[2]]
        self.moveto(coord, 0.0, 1.0)
        time.sleep(1)
        self.catch(wide)
        coord = [fromXYZ[0], fromXYZ[1], height]
        self.moveto(coord, 0.0, 1.0)
        time.sleep(1)
        
        coord = [toXYZ[0], toXYZ[1], height]
        self.moveto(coord, 0.0, 1.0)
        time.sleep(1)
        
        coord = [toXYZ[0], toXYZ[1], toXYZ[2]]
        self.moveto(coord, 0.0, 1.0)
        time.sleep(1)
        
        self.release()
        coord = [toXYZ[0], toXYZ[1], height]
        self.moveto(coord, 0.0, 1.0)
        time.sleep(1)

    def resetArm(self):
        p = Command()
        p.fdat = (c_float * 6)(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        p.idat[0] = 3
        self.ser.flushInput()
        self.ser.write(p.raw)
        while (self.ser.in_waiting == 0):
            pass
        backstr = self.ser.read(5)
        return backstr

    def moveto(self, coordinate=(30.0, 0.0, 2.0), spinAngle=0.0, vel=10.0):
        p = Command()
        p.fdat = (c_float * 6)(1.0, coordinate[0], coordinate[1], coordinate[2], spinAngle, vel)
        p.idat[0] = 1
        self.ser.flushInput()
        self.ser.write(p.raw)
        while(self.ser.in_waiting == 0):
            pass
        backstr = self.ser.read(5)
        return backstr

    def catch(self, wide=4.7):
        p = Command()
        p.fdat = (c_float * 6)(1.0, wide, 0.0, 0.0, 0.0, 0.0)
        p.idat[0] = 2
        self.ser.flushInput()
        self.ser.write(p.raw)
        while (self.ser.in_waiting == 0):
            pass
        backstr = self.ser.read(5)
        return backstr

    def release(self):
        p = Command()
        p.fdat = (c_float * 6)(1.0, 4.7, 0.0, 0.0, 0.0, 0.0)
        p.idat[0] = 2
        self.ser.flushInput()
        self.ser.write(p.raw)
        while (self.ser.in_waiting == 0):
            pass
        backstr = self.ser.read(5)
        return backstr

if __name__ == '__main__':
    arm = RobotArm('COM11')
    coordinate = (30.0, 0.0, 2.0)
    spinAngle = 0.0
    vel = 10.0
    ret = arm.moveto(coordinate, spinAngle , vel)