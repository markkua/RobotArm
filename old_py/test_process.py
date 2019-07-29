import pyrealsense2 as rs
import numpy as np
import cv2


def image_process():
    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.visual_preset, 4)

    # 创建align对象
    align_to = rs.stream.color
    align = rs.align(align_to)

    try:
        while True:
            frames = pipeline.wait_for_frames()

            # 对齐
            align_frames = align.process(frames)

            depth_frame = align_frames.get_depth_frame()
            color_frame = align_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # color_image = np.asanyarray(color_frame.get_data())
            # cv2.imshow('rgb', color_image)

            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            dec = rs.decimation_filter()
            dec.set_option(rs.option.filter_magnitude, 1)
            depth_frame_pro = dec.process(depth_frame)

            depth2disparity = rs.disparity_transform()
            depth_frame_pro = depth2disparity.process(depth_frame_pro)

            spat = rs.spatial_filter()
            # 启用空洞填充,5为填充所有零像素
            spat.set_option(rs.option.holes_fill, 5)
            depth_frame_pro = spat.process(depth_frame_pro)

            temp = rs.temporal_filter()
            depth_frame_pro = temp.process(depth_frame_pro)

            disparity2depth = rs.disparity_transform(False)
            depth_frame_pro = disparity2depth.process(depth_frame_pro)

            depth_image_pro = np.asanyarray(depth_frame_pro.get_data())
            depth_colormap_pro = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_pro, alpha=0.03), cv2.COLORMAP_JET)
            
            images = np.hstack((depth_colormap, depth_colormap_pro))
            cv2.imshow('depth_images', images)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pipeline.stop()

    return depth_image_pro
