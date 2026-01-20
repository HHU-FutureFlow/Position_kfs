import cv2
import numpy as np
import pyrealsense2 as rs
import time


ArUcoDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


pipeline = rs.pipeline()

# 创建配置对象
config = rs.config()

# 启用彩色和深度流
config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

# 开始流传输
profile = pipeline.start(config)

sensor = profile.get_device().first_depth_sensor()
sensor.set_option(rs.option.emitter_enabled, 0) # 0代表关闭，1代表开启
sensor.set_option(rs.option.laser_power, 0)


dist_coeffs = np.zeros((1, 5))

try:
    while True:

        fps_start_time = time.time()
        frame_count = 0

        #等待一组帧（深度和彩色）
        frames = pipeline.wait_for_frames()

        infrared_frame = frames.get_infrared_frame()

        infrared_intrin = infrared_frame.profile.as_video_stream_profile().intrinsics  # 获取相机内参

        camera_matrix = np.array([
            [infrared_intrin.fx, 0, infrared_intrin.ppx],
            [0, infrared_intrin.fy, infrared_intrin.ppy],
            [0, 0, 1]
        ], dtype=np.float32)


        # 转换为numpy数组 Cir:低运行效率段，考虑优化
        infrared_image = np.asanyarray(infrared_frame.get_data())

        # 使用OpenCV的ArUco检测器
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(ArUcoDictionary, parameters)

        corners = []
        rejected = 0

        # 检测标记
        corners, ids, rejected = detector.detectMarkers(infrared_image)

        if ids is not None:
            # 绘制检测结果
            image_with_markers = infrared_image.copy()
            cv2.aruco.drawDetectedMarkers(image_with_markers, corners, ids)

            # 计算位姿
            if len(corners) > 0:
                marker_size = 0.1  # 5cm
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, camera_matrix, dist_coeffs
                )

            # 显示检测结果
            for i, (corner, id_val, rvec, tvec) in enumerate(zip(corners, ids, rvecs, tvecs)):
                print(f" 标记 ID {id_val[0]}:")
                print(f"   位置: x={tvec[0][0]:.3f}m, y={tvec[1][0]:.3f}m, z={tvec[2][0]:.3f}m")

            frame_count += 1
            fps_end_time = time.time()
            fps = frame_count / (fps_end_time - fps_start_time)
            cv2.putText(
                image_with_markers,
                f"FPS: {fps:.1f}",
                (10, 30),  # 文字位置（左上角）
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 字体大小
                (0, 255, 0),  # 颜色（绿色）
                2  # 线条粗细
            )

        cv2.imshow('Depth',image_with_markers)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
