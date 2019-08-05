# -*- encoding: utf-8 -*-

import cv2

from realsenseCamera_old import RealsenseCamera
from ToolDetect import *
from SerialPart import *
import time
# from speech.VoiceRecognition import *


if __name__ == '__main__':
    camera = RealsenseCamera()
    model = create_model('mask_rcnn_tools_0030.h5')
 
    image = camera.get_color_image_from_frames(camera.get_aligned_frames())
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cv2.imshow('image', image)
    cv2.waitKey(1)

    start_time = time.time()
    r = model.detect([image], verbose=0)[0]
    
    print('time=', time.time() - start_time)
    
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    
    cout, list = calculate(r)
    print(cout,list)
    