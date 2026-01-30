import pyrealsense2 as rs
import numpy as np
from struck.strukt import *
from util import DoubleParameterFilter, SingleParameterFilter


def Start_stream():
    pipeline = rs.pipeline()

    # 创建配置对象
    config = rs.config()

    # 启用彩色和深度流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始流传输
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    return align, pipeline

def align_image(align, pipeline):
    # 等待一组帧（深度和彩色）
    frames = pipeline.wait_for_frames()

    # 对齐帧
    aligned_frames = align.process(frames)

    # 获取对齐后的帧
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics  # 获取深度参数（像素坐标系转相机坐标系会用到）
    color_intrin = color_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

    # 转换为numpy数组 Cir:低运行效率段，考虑优化
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    D435i = CameraStream(aligned_depth_frame, depth_intrin, color_intrin, depth_image, color_image)
    return D435i

def getpoint(results, current_result, last_result,depth_frame):
    if results[0].boxes is not None:
        if len(results[0].boxes) > 0:
            box_xywh = results[0].boxes.xywh[0].cpu().numpy()  # 单目标取第一个框，转numpy数组
            current_result.x = int(box_xywh[0])  # 中心x，强制转整数
            current_result.y = int(box_xywh[1])  # 中心y，强制转整数
            current_result.w = int(box_xywh[2])  # 宽度，强制转整数
            current_result.h = int(box_xywh[3])  # 高度，强制转整数

            final_results = DoubleParameterFilter.initialize(last_result, current_result)
            center_x, center_y, width, high = final_results.x, final_results.y, final_results.w, final_results.h
            class_ids = results[0].boxes.cls.cpu().numpy()

            if class_ids[0] == 0:  # Cir： 深度可以取周边几个像素平均值，如果相差过大说明识别错误，重新识别
                Pixel_target_x, Pixel_target_y = center_x, int(center_y + (high * 0.52))
                color_pixel = [Pixel_target_x, Pixel_target_y]
                if Pixel_target_x > 640 or Pixel_target_y > 480:
                    color_pixel = [0, 0]
                Camera_target_z = depth_frame.get_distance(color_pixel[0], color_pixel[1])

            elif class_ids[0] == 1:  # Cir： 深度可以取周边几个像素平均值，如果相差过大说明识别错误，重新识别
                Pixel_target_x, Pixel_target_y = int(center_x - (width * 0.15)), int(center_y + (high * 1))
                color_pixel = [Pixel_target_x, Pixel_target_y]
                if Pixel_target_x > 640 or Pixel_target_y > 480:
                    color_pixel = [0, 0]
                Camera_target_z = depth_frame.get_distance(color_pixel[0], color_pixel[1])

            elif class_ids[0] == 2:  # Cir： 深度可以取周边几个像素平均值，如果相差过大说明识别错误，重新识别
                Pixel_target_x, Pixel_target_y = center_x, int(center_y + (high * 0.9))
                color_pixel = [Pixel_target_x, Pixel_target_y]
                if Pixel_target_x >= 640 or Pixel_target_y >= 480:
                    color_pixel = [0, 0]
                Camera_target_z = depth_frame.get_distance(color_pixel[0], color_pixel[1])

            last_result = final_results
            Pixel_position = point(Pixel_target_x, Pixel_target_y, Camera_target_z)
            return Pixel_position, last_result, final_results

        else:
            # 容错：无有效检测目标（数组为空）
            No_final = xywh()
            print("提示：未检测到有效目标，跳过目标坐标提取")
            return point(), last_result, No_final
    else:
        # 容错：无检测结果容器（boxes为None）
        No_final = xywh()
        print("提示：无任何检测结果，boxes为None")
        # 应用颜色映射到深度图像（用于可视化）
        return point, last_result, No_final