from ultralytics import YOLO
import cv2

def YoloDetect(model_path, stream_url, conf_threshold=0.5 ):
    """
       YOLO 模型实时检测直播流
       :param model_path: 训练好的 YOLO 模型路径（.pt 文件）
       :param stream_url: 直播流地址（RTSP/RTMP/HTTP-FLV 等）
       :param conf_threshold: 置信度阈值（过滤低置信度检测结果）
       """
    model = YOLO(model_path)
    print(f"成功加载模型{model_path}")

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError(f"无法加载直播流，可能地址拼写错误")

    fps_start_time = time.time()
    frame_count = 0
    print(f"开启检测，按‘q’退出")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("获取帧失败")
            break

        result = model(frame,conf=conf_threshold)

        anotated