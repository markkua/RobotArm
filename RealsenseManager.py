# -*- encoding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np
import cv2
from typing import List
from ConsolePrinter import Printer


class RealsenseManager:
	"""
	用于设置、处理Realsense相机的类，提供frames，
	"""
	
	def __init__(self, color_resolution=(1920, 1080), depth_resolution=(1280, 720)):
		try:
			self.pipeline = rs.pipeline()
			self.config = rs.config()
			self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, 30)
			self.config.enable_stream(rs.stream.color, color_resolution[0], color_resolution[1], rs.format.bgr8, 30)
			
			# 配置文件
			self.profile = self.pipeline.start(self.config)
			
			# 深度传感器
			self.depth_sensor = self.profile.get_device().first_depth_sensor()
			self.depth_sensor.set_option(rs.option.visual_preset, 4)
			# 深度标尺
			self.depth_scale = self.depth_sensor.get_depth_scale()
			print('depth_scale=', self.depth_scale)
			
			# 创建align对象
			self.align_to = rs.stream.color
			self.align = rs.align(self.align_to)
			
			Printer.print("Camera ready", Printer.green)
		except RuntimeError as e:
			Printer.print("Camera init fail: " + e.__str__(), Printer.red)
		
	def get_aligned_frames(self):
		"""
        获取对齐的帧, 深度向颜色对齐, 失败返回None
		:return: 对齐后的帧(frames)，获取失败返回None
		"""
		try:
			frames = self.pipeline.wait_for_frames()
			
			# 深度向颜色对齐
			align_frames = self.align.process(frames)
			
			return align_frames
		except cv2.error as e:
			Printer.print('Get aligned frames error: %s' % e.__str__(), Printer.red)
			return None
	
	def get_coor_in_Camera_system(self, align_frames, pixel_coor):
		"""
		获取指定像素点[x, y]在相机坐标系下的三维坐标[X, Y, Z]
		:param align_frames: 对齐的帧
		:param pixel_coor: 目标像素的坐标
		:return:
		"""
		# 获得对齐后的深度帧
		align_depth_frame = align_frames.get_depth_frame()
		color_frame = align_frames.get_color_frame()
		
		if not color_frame or not align_depth_frame:
			return None
		
		# 获得颜色帧的内参
		color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
		
		align_depth_image = np.asanyarray(align_depth_frame.get_data())
		# 长度单位为cm
		align_depth_value = align_depth_image[pixel_coor[0], pixel_coor[1]] * self.depth_scale * 100  # 单位转换，100为cm，1为m
		
		# 输入传感器内部参数、点的像素坐标、对应的深度值，输出点的三维坐标。
		result_coor = rs.rs2_deproject_pixel_to_point(color_intrin, pixel_coor, align_depth_value)
		
		# 判断结果是否异常
		if [0, 0, 0] == result_coor:
			Printer.print("Coor error, maybe too near", Printer.red)
			return None
		return result_coor
	
	@staticmethod
	def get_color_image_from_frames(frames):
		color_frame = frames.get_color_frame()
		color_image = np.asanyarray(color_frame.get_data())
		return color_image
	
	@staticmethod
	def get_depth_image_from_frames(frames):
		depth_frame = frames.get_depth_frame()
		depth_image = np.asanyarray(depth_frame.get_data())
		return depth_image
		
	@staticmethod
	def _filter_depth_frame(depth_frame):
		"""
		滤波器，用于获取坐标前的深度图像处理
		:param depth_frame: 深度帧
		:return: 滤波后的深度帧
		"""
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
		
		# depth_image_pro = np.asanyarray(depth_frame_pro.get_data())
		# depth_colormap_pro = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_pro, alpha=0.03), cv2.COLORMAP_JET)

		return depth_frame_pro
		
	def view_color_image(self):
		"""简单查看一下RGB工作状态"""
		try:
			while True:
				frames = self.pipeline.wait_for_frames()
				
				color_frame = frames.get_color_frame()
			
				color_image = np.asanyarray(color_frame.get_data())
				
				# Show images
				cv2.namedWindow('Color image', cv2.WINDOW_AUTOSIZE)
				cv2.imshow('Color image', color_image)
				
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q') or key == 27:
					cv2.destroyAllWindows()
					Printer.print('Quit View', Printer.green)
					return
		except cv2.error as e:
			Printer.print('View color image error: %s' % e.__str__(), Printer.red)
			
	def view_aligned_filled_images(self):
		"""简单查看对齐的RGB和深度图像的状态"""
		try:
			while True:
				align_frames = self.get_aligned_frames()
				color_frame = align_frames.get_color_frame()
				color_image = np.asanyarray(color_frame.get_data())
				
				depth_frame_pro = self._filter_depth_frame(align_frames.get_depth_frame())
				depth_image_pro = np.asanyarray(depth_frame_pro.get_data())
				depth_colormap_pro = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_pro, alpha=0.03), cv2.COLORMAP_JET)
				
				images = np.hstack((color_image, depth_colormap_pro))
			
				# Show images
				cv2.namedWindow('Color and aligned filled depth', cv2.WINDOW_AUTOSIZE)
				cv2.imshow('Color and aligned filled depth', images)
				
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q') or key == 27:
					cv2.destroyAllWindows()
					Printer.print('Quit View', Printer.green)
					return
			
		except cv2.error as e:
			Printer.print('View aligned filled image error: %s' % e.__str__(), Printer.red)
			
	def test_coor(self):
		point = [320, 240]
		while True:
			aligned_frames = self.get_aligned_frames()
			
			depth_frame = aligned_frames.get_depth_frame()
			color_frame = aligned_frames.get_color_frame()
			color_image = np.asanyarray(color_frame.get_data())
			
			cv2.circle(color_image, (point[0], point[1]), 3, [0, 0, 255])
			
			# Show images
			cv2.namedWindow('Color image', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('Color image', color_image)
			
			coor = self.get_coor_in_Camera_system(aligned_frames, point)
			print(coor)
			# depth = (depth_frame.get_distance(point[0], point[1])) * 100
			# print("depth=", depth)
			# print("delta depth=(cm)", depth - coor[2])
			
			key = cv2.waitKey(500)
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				Printer.print('Quit View', Printer.green)
				break
			elif key == ord('a'):
				point[0] -= 1
			elif key == ord('d'):
				point[0] += 1
			elif key == ord('s'):
				point[1] += 1
			elif key == ord('w'):
				point[1] -= 1


if __name__ == '__main__':
	realsense = RealsenseManager(color_resolution=(640, 480), depth_resolution=(640, 360))
	realsense.test_coor()
