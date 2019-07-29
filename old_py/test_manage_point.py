# -*- encoding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()

config = rs.config()
# 深度和颜色流的不同分辨率
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 配置文件
profile = pipeline.start(config)

# 获取深度传感器的深度标尺
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("depth scale is ", depth_scale)

# 创建对齐对象
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        frames = pipeline.wait_for_frames()

        # 获得图像帧
        color_frame = frames.get_color_frame()

        # 深度向颜色对齐
        align_frames = align.process(frames)
        # 获得对齐后的深度帧
        align_depth_frame = align_frames.get_depth_frame()

        if not color_frame or not align_depth_frame:
            continue

        # 获得颜色帧的内参
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics

        # 要获得的像素点
        color_pixel = [100, 100]

        align_depth_image = np.asanyarray(align_depth_frame.get_data())
        # 长度单位为cm
        align_depth_value = align_depth_image[color_pixel[0], color_pixel[1]] * depth_scale * 10
        print("align depth value is", align_depth_value)

        # 输入传感器内部参数、点的像素坐标、对应的深度值，输出点的三维坐标。
        color_point = rs.rs2_deproject_pixel_to_point(color_intrin, color_pixel, align_depth_value)

        print(color_point)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
