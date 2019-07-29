# -*- encoding: utf-8 -*-

import cv2
import numpy as np
from pyzbar.pyzbar import decode
from RealsenseManager import RealsenseManager


def pre_process(image):
 
    # 高斯滤波
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # b, g, r = cv2.split(blur_image)  # bgr
    
    hsv_image = cv2.cvtColor(blur_image, cv2.COLOR_RGB2HLS)
    
    green_hsv = np.asarray([
        [37, 43, 46],  # lower
        [77, 255, 255]  # upper
    ])
    
    blue_hsv = np.asarray([
        [100, 43, 46],
        [170, 255, 255]
    ])

    # mask = cv2.inRange(hsv_image, lowerb=green_hsv[0], upperb=green_hsv[1])
    mask = cv2.inRange(hsv_image, lowerb=blue_hsv[0], upperb=blue_hsv[1])
    b_image = cv2.bitwise_and(blur_image, blur_image, mask=mask)
    
    # b_image = np.floor(blur_image * 0.3 + b_image)
    # b_image = np.asarray(b_image, dtype=np.uint8)

    point = [320, 240]
    cv2.circle(image, (point[0], point[1]), 3, [0, 0, 255])
    print("center point HSV:", hsv_image[point[0], point[1]])
    
    # 给图加标签
    cv2.putText(image, "Image", (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255))
    cv2.putText(b_image, "B-image", (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (255, 255, 255))

    result = np.hstack((image, b_image))
    
    return result
    
    
    # binary_image = cv2.threshold(blur_image, 100, 255, cv2.THRESH_BINARY)
    
    # cv2.imshow("二值化", binary_image)
    # cv2.waitKey(0)
    #
    # element = cv2.getStructuringElement(2, [7, 7])
    # for i in range(10):
    #     erode_image = cv2.erode(image, element)
    #
    # cv2.imshow("腐蚀", )
    
    
if __name__ == '__main__':
    realsense = RealsenseManager(color_resolution=(640, 480), depth_resolution=(640, 360))
    # realsense.view_color_image()
    
    flag = True
    while flag:
        
        image = realsense.get_color_image_from_frames(realsense.get_aligned_frames())
        
        image = pre_process(image)
        
        cv2.imshow("QRcode", image)
        key = cv2.waitKey(1)
        if ord('q') == key or 27 == key:
            flag = False
    
    