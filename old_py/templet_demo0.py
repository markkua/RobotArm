# -*- encoding: utf-8 -*-

import cv2
import numpy as np


def mathch_img(target, template_imgname, threshold):
    img_rgb = cv2.imread(target)
    # 创建一个原始图像的灰度版本，所有操作在灰度版本中处理，然后在RGB图像中使用相同坐标还原
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # 模板
    template = cv2.imread(template_imgname, 0)
    w, h = template.shape[:2]
    
    # 使用matchTemplate对原始灰度图像和图像模板进行匹配
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
    loc = np.where(res >= threshold)
    print(loc)
    
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 2)
    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    target_imgname = "data/img.jpg"
    template_imgname = "data/template.jpg"
    cv2.imshow('template', cv2.imread(template_imgname, cv2.IMREAD_COLOR))
    cv2.waitKey(0)
    
    value = 0.5
    mathch_img(target_imgname, template_imgname, value)
