from ultralytics import YOLO
import pyrealsense2 as rs
import numpy as np
import cv2
import time


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

model = YOLO(r'D:\Position\re_weapon.pt')

try:
    while True:

        fps_start_time = time.time()
        frame_count = 0

        #等待一组帧（深度和彩色）
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

        results = model(color_image, conf=0.5)

        pixel_target_x, Pixel_target_y, Camera_target_z = 0, 0, 0.0
        color_pixel = [0,0]
        class_id = -1  # 用-1标记无有效目标
        boxes_xywh = np.array([])  # 初始化空数组
        class_ids = np.array([])  # 初始化空数组

        if results[0].boxes is not None:
            # 提取所有目标的边界框（xyxy格式）、置信度、类别ID
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()  # 转为numpy数组（推荐cpu()避免GPU数据报错）
            class_ids = results[0].boxes.cls.cpu().numpy()

            if len(boxes_xywh) > 0 and len(class_ids) > 0:
                center_x, center_y, high = int(boxes_xywh[0, 0]), int(boxes_xywh[0, 1]), int(boxes_xywh[0, 3])

                if class_ids[0] == 0: #Cir： 深度可以取周边几个像素平均值，如果相差过大说明识别错误，重新识别
                    Pixel_target_x, Pixel_target_y = center_x, int(center_y + (high*0.75))
                    color_pixel = [Pixel_target_x, Pixel_target_y]
                    Camera_target_z = aligned_depth_frame.get_distance(color_pixel)

            else:
                # 容错：无有效检测目标（数组为空）
                print("提示：未检测到有效目标，跳过目标坐标提取")
        else:
            # 容错：无检测结果容器（boxes为None）
            print("提示：无任何检测结果，boxes为None")
            # 应用颜色映射到深度图像（用于可视化）


        #Cir: 未考虑畸变参数，所以使用以下函数不确定是否像素对齐irhongwai Normalize_target_x, Normalize_target_y =

        Camera_position = rs.rs2_deproject_pixel_to_point(depth_intrin, color_pixel , Camera_target_z)


        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET)

        annotated_frame = results[0].plot()

        point_coords = (int(Camera_position[0]), int(Camera_position[1]))
        radius = 1
        color = (0, 0, 255)
        thickness = -1

        cv2.circle(annotated_frame, point_coords, radius, color, thickness)

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
