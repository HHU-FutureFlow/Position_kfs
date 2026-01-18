from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import time

def RealsenseConfig():
    # 创建管道
    pipeline = rs.pipeline()

    # 创建配置对象
    config = rs.config()

    # 启用彩色和深度流
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # 开始流传输
    profile = pipeline.start(config)

    fps_start_time = time.time()

    # 获取深度传感器的深度标尺（单位：米）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale: {depth_scale}")

    align_to = rs.stream.color
    align = rs.align(align_to)


    # 等待一组帧（深度和彩色）
    frames = pipeline.wait_for_frames()

    # 对齐帧
    aligned_frames = align.process(frames)

    # 获取对齐后的帧
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    # 转换为numpy数组 Cir:低运行效率段，考虑优化
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return depth_image,color_image,fps_start_time


#预留接口，从yolo模型中获取目标框中心点，提取中心点坐标，再进行映射
def GlobalDepthVisualize(Depthimage, Colorimage):
    # 应用颜色映射到深度图像（用于可视化）
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(Depthimage, alpha=0.03),
        cv2.COLORMAP_JET)

    return depth_colormap


def YoloDetect(model, frames, conf_threshold=0.5):
    """
       YOLO 模型实时检测直播流
       :param model_path: 训练好的 YOLO 模型路径（.pt 文件）
       :param conf_threshold: 置信度阈值（过滤低置信度检测结果）
       """
    results = model(frames, conf=conf_threshold)
    return results

def drawFrame(yolo_result, depthframe, target, fps_start_time, frameCount=0):
    annotated_frame = yolo_result[0].plot()

    point_coords = (target[0],target[1])
    radius = 3
    color = (0,0,255)
    thickness = -1

    cv2.circle(annotated_frame, point_coords, radius, color, thickness)

    frameCount += 1
    fps_end_time = time.time()
    fps = frameCount / (fps_end_time - fps_start_time)
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),  # 文字位置（左上角）
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # 字体大小
        (0, 255, 0),  # 颜色（绿色）
        2  # 线条粗细
    )

    cv2.imshow('Color', annotated_frame)


def positiondetect(depthframe, yolo_results):
    if yolo_results[0].boxes is not None:
        # 提取所有目标的边界框（xyxy格式）、置信度、类别ID
        boxes_xywh = yolo_results[0].boxes.xywh.cpu().numpy() # 转为numpy数组（推荐cpu()避免GPU数据报错）
        class_ids = yolo_results[0].boxes.cls.cpu().numpy()

        center_x, center_y, high= int(boxes_xywh[0,0]), int(boxes_xywh[0,1]), int(boxes_xywh[0,3])

        if class_ids[0] == 1:
            target_x, target_y = center_x, int(center_y+(high/4))
            target_z = float(depthframe[target_y,target_x])
        return class_ids, target_x, target_y, target_z

    else:
        return 0







