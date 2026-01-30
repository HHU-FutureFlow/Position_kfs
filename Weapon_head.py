from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import time
from util import draw, Realsence
from struck.strukt import *

align, pipeline = Realsence.Start_stream()

model = YOLO(r'D:\Position\best.pt')
last_result = xywh()
current_result = xywh()

try:
    while True:

        fps_start_time = time.time()
        frame_count = 0

        D435i = Realsence.align_image(align, pipeline)
        depth_frame = D435i.depth_frame
        color_image = D435i.color_image
        depth_image = D435i.depth_image
        depth_intrin = D435i.depth_intrin
        color_intrin = D435i.color_intrin

        results = model(color_image, conf=0.5)

        Pixel_target_x, Pixel_target_y, Camera_target_z = 0, 0, 0.0
        color_pixel = [0,0]
        class_id = -1  # 用-1标记无有效目标
        boxes_xywh = np.array([])  # 初始化空数组
        class_ids = np.array([])  # 初始化空数组

        Pixel_position,last_result, final_result = Realsence.getpoint(results,current_result,last_result,depth_frame)


        Camera_target_z = round(Pixel_position.z,3)

        #Cir: 未考虑畸变参数，所以使用以下函数不确定是否像素对齐irhongwai Normalize_target_x, Normalize_target_y =

        Camera_position = rs.rs2_deproject_pixel_to_point(depth_intrin, color_pixel , Camera_target_z)


        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)

        annotated_frame = draw.drawpoint(color_image, Pixel_position)

        draw.draw_box(annotated_frame, final_result)



        cv2.putText(
            annotated_frame,
            f"Depth: {Camera_target_z}",
            (10, 60),  # 文字位置（左上角）
            cv2.FONT_HERSHEY_SIMPLEX,
            1,  # 字体大小
            (0, 0, 255),  # 颜色（绿色）
            2  # 线条粗细
        )

        frame_count += 1
        fps_end_time = time.time()
        fps = frame_count / (fps_end_time - fps_start_time)
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
        cv2.imshow('Depth',depth_colormap)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
